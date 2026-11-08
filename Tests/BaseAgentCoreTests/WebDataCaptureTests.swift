import XCTest
import Foundation
@testable import BaseAgentCore

/// web_scan 动态数据接口直取：候选 URL 筛选 + JSON/HTML 精简。
final class WebDataCaptureTests: XCTestCase {

    private let base = URL(string: "https://jobs.example.com/list")!

    private func data(_ json: String) -> Data { Data(json.utf8) }

    // MARK: - 候选 URL

    func testCandidateURLsPreferRuntimeCapturedURLWithParameters() {
        let captured = [WebRenderer.CapturedRequest(
            url: "/api/SearchIndex/GetSearch?SearchText=job,*|^|Sorting=posted_On",
            status: 0, body: "")]
        let urls = WebDataCapture.candidateURLs(
            captured: captured,
            endpointHints: ["/api/SearchIndex/Search"], baseURL: base)
        XCTAssertEqual(urls.first?.path, "/api/SearchIndex/GetSearch")
        XCTAssertTrue(urls.first?.query?.contains("SearchText=job") ?? false,
                      "页面运行时拼好的查询参数必须原样保留")
        XCTAssertTrue(urls.contains { $0.path == "/api/SearchIndex/Search" })
    }

    func testCandidateURLsRejectCrossOriginAndStaticAssets() {
        let captured = [
            WebRenderer.CapturedRequest(url: "https://tracker.other.com/api/x",
                                        status: 200, body: "{}"),
            WebRenderer.CapturedRequest(url: "/assets/app.js", status: 200, body: ""),
            WebRenderer.CapturedRequest(url: "/images/logo.png", status: 200, body: ""),
        ]
        let urls = WebDataCapture.candidateURLs(
            captured: captured, endpointHints: ["/api/jobs"], baseURL: base)
        XCTAssertEqual(urls.map(\.path), ["/api/jobs"])
    }

    // MARK: - JSON 精简

    private func azureJSON(results: String, count: Int = 3405) -> String {
        """
        {"Facets":{"position_Subject":[{"Value":"Teacher"}]},
         "Results":\(results),
         "Count":\(count), "PageSize":25,
         "SearchFilters":{"ModuleName":"job","SearchText":"*",
                          "CurrentPage":0,"Sorting":"posted_On"}}
        """
    }

    func testSummarizeJSONListsRecordsWithTotalAndPaging() {
        let results = """
        [{"Document":{"position_Subject":"Elementary Teacher",
                      "posted_On":"2026-11-07T00:00:00+00:00",
                      "employer_Name":"Keewatin-Patricia DSB",
                      "city":"Kenora","job_Code":"4041442"}},
         {"Document":{"position_Subject":"Education Assistant",
                      "posted_On":"2026-11-07T00:00:00+00:00",
                      "employer_Name":"Grasslands Public Schools",
                      "city":"Brooks","job_Code":"4071206"}}]
        """
        let text = WebDataCapture.summarize(
            data: data(azureJSON(results: results)),
            contentType: "application/json", label: "/api/SearchIndex/GetSearch")
        XCTAssertNotNil(text)
        XCTAssertTrue(text?.contains("共 3405 条") ?? false)
        XCTAssertTrue(text?.contains("本页 2 条") ?? false)
        XCTAssertTrue(text?.contains("Elementary Teacher") ?? false)
        XCTAssertTrue(text?.contains("2026-11-07") ?? false)
        XCTAssertTrue(text?.contains("Keewatin-Patricia DSB") ?? false)
        XCTAssertTrue(text?.contains("分页参数（响应自带）") ?? false)
        XCTAssertTrue(text?.contains("CurrentPage=0") ?? false)
        XCTAssertTrue(text?.contains("当天记录可能还有后续页") ?? false)
    }

    func testSummarizePrefersResultsOverLargerMetadataArray() {
        let json = """
        {"Results":[{"Document":{"position_Subject":"Only Real Job"}}],
         "FacetControls":[{"FieldName":"a"},{"FieldName":"b"},
                          {"FieldName":"c"},{"FieldName":"d"},{"FieldName":"e"}]}
        """
        let text = WebDataCapture.summarize(
            data: data(json), contentType: "application/json", label: "/api/x")
        XCTAssertTrue(text?.contains("Only Real Job") ?? false)
        XCTAssertFalse(text?.contains("FieldName") ?? true)
        XCTAssertTrue(text?.contains("本页 1 条") ?? false)
    }

    func testSummarizeDemotesInternalUUIDFields() {
        let json = """
        {"Results":[{"Document":{
            "position_Subject":"Math Teacher", "posted_On":"2026-11-07",
            "employer_Name":"Some Board", "employer_Id":"2cdb32ae-0695-44f9-86a2-71516d6d8836",
            "city":"Kenora", "province":"ON", "job_Type":"Permanent",
            "job_Category":"Teaching", "fte":"1.0", "status":"ACTIVE",
            "job_Description":"A very long description of the open position"}}]}
        """
        let text = WebDataCapture.summarize(
            data: data(json), contentType: "application/json", label: "/api/x") ?? ""
        XCTAssertTrue(text.contains("Math Teacher"))
        XCTAssertFalse(text.contains("2cdb32ae"), "内部 UUID 不该挤掉可读字段")
        XCTAssertTrue(text.contains("Kenora"))
    }

    func testSummarizeRejectsErrorAndNullShells() {
        let bodies = [
            "{\"Message\":\"No HTTP resource was found\",\"MessageDetail\":\"No action\"}",
            "{\"Facets\":null,\"Results\":null,\"Count\":null,\"ErrorId\":\"abc\"}",
            "{\"Results\":[],\"Count\":0}",
            "{\"Facets\":{\"x\":[{\"Value\":\"Teacher\"}]},\"Count\":10}",
        ]
        for body in bodies {
            XCTAssertNil(WebDataCapture.summarize(
                data: data(body), contentType: "application/json", label: "/api/x"),
                "错误体/空壳不得被当数据：\(body.prefix(60))")
        }
    }

    func testSummarizeTopLevelArray() {
        let json = """
        [{"title":"Job A","posted_On":"2026-11-07"},
         {"title":"Job B","posted_On":"2026-11-07"}]
        """
        let text = WebDataCapture.summarize(
            data: data(json), contentType: "application/json", label: "/api/jobs")
        XCTAssertTrue(text?.contains("Job A") ?? false)
        XCTAssertTrue(text?.contains("Job B") ?? false)
    }

    // MARK: - HTML 响应

    func testSummarizeHTMLOnlyWhenSubstantive() {
        let page = "<html><body><h1>Jobs</h1><p>"
            + String(repeating: "招聘信息 ", count: 60) + "</p></body></html>"
        let text = WebDataCapture.summarize(
            data: data(page), contentType: "text/html", label: "/ajax/list")
        XCTAssertTrue(text?.contains("招聘信息") ?? false)

        let error = "<html><body><p>No HTTP resource was found that matches "
            + String(repeating: "the request URI ", count: 30) + "</p></body></html>"
        XCTAssertNil(WebDataCapture.summarize(
            data: data(error), contentType: "text/html", label: "/ajax/list"))
    }

    // MARK: - 只读分页合并

    func testPageSpecKeepsResponseFiltersAndPicksReadOnlyEndpoint() throws {
        let json = azureJSON(results: """
        [{"Document":{"position_Subject":"Job A",
                      "posted_On":"2026-11-07T00:00:00+00:00"}}]
        """)
        let root = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: data(json)) as? [String: Any])
        let getURL = URL(string: "https://jobs.example.com/api/SearchIndex/GetSearch?SearchText=job,*")!
        let postURL = URL(string: "https://jobs.example.com/api/SearchIndex/Search")!
        let writeURL = URL(string: "https://jobs.example.com/api/SearchIndex/AddJobFilter")!
        // 真实页面上这些干扰端点排在真正的 Search 之前：不同模块的 getcountry、
        // 同模块但语义是写/建议的 PostSuggest，都不能被选作翻页端点。
        let noiseURL = URL(string: "https://jobs.example.com/api/searchip/getcountry?ipaddress=")!
        let suggestURL = URL(string: "https://jobs.example.com/api/SearchIndex/PostSuggest")!
        let plan = WebDataCapture.pageSpec(
            root: root,
            candidates: [getURL, noiseURL, suggestURL, writeURL, postURL],
            exclude: getURL)
        XCTAssertEqual(plan?.endpoint, postURL)
        XCTAssertEqual(plan?.pageKey, "CurrentPage")
        XCTAssertEqual(plan?.startPage, 0)
        XCTAssertEqual(plan?.dateValue, "2026-11-07")
        XCTAssertEqual(plan?.form["Sorting"], "posted_On")
        XCTAssertEqual(plan?.form["ModuleName"], "job")
    }

    func testReadOnlyEndpointFilterRejectsWriteEndpoints() {
        let read = URL(string: "https://jobs.example.com/api/SearchIndex/Search")!
        let write = URL(string: "https://jobs.example.com/api/SearchIndex/AddJobFilter")!
        XCTAssertTrue(WebDataCapture.isReadOnlySearchEndpoint(read))
        XCTAssertFalse(WebDataCapture.isReadOnlySearchEndpoint(write))
    }

    func testMergedSummaryMarksSameDayComplete() {
        let records: [[String: Any]] = (1...54).map { index in
            ["Document": ["position_Subject": "Job \(index)",
                          "posted_On": "2026-11-07T00:00:00+00:00"]]
        }
        let text = WebDataCapture.summary(
            records: records, total: 3405,
            label: "/api/SearchIndex/GetSearch（已自动合并当天分页）",
            pageInfo: ["CurrentPage": "0"], allDatesCollected: true, maxChars: 16_000)
        XCTAssertTrue(text?.contains("本页 54 条") ?? false)
        XCTAssertTrue(text?.contains("Job 54") ?? false)
        XCTAssertFalse(text?.contains("当天记录可能还有后续页") ?? true)
    }
}
