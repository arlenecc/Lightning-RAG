import SwiftUI
import AppKit
import UniformTypeIdentifiers
import BaseAgentCore

/// MOBI / AZW3 扫描版电子书阅读器。
///
/// 解析层只返回「图片数组 + 原始字节」，这里用 SwiftUI `LazyVStack` 做
/// 懒加载（macOS 无 UIKit 的 UICollectionView / UIPageViewController，
/// LazyVStack + 页缓存窗口是等价实现）：只为当前页 ±1 保留 CGImage，
/// 翻过一页立刻释放离开窗口的页图，避免整本书图片常驻内存。
@MainActor
struct MobiScanReaderView: View {
    let model: AppViewModel

    @State private var pages: [MobiScannedPage] = []
    @State private var stitchedRanges: [Range<Int>] = []
    @State private var currentIndex = 0
    /// 页图缓存窗口：仅当前页与相邻页；其它页在滚动离开时立即移除。
    @State private var imageCache: [Int: CGImage] = [:]
    @State private var status = "尚未打开 MOBI/AZW3 文件"
    @State private var sourceURL: URL?

    var body: some View {
        VStack(spacing: 0) {
            toolbar
            Divider()
            content
        }
    }

    private var toolbar: some View {
        HStack(spacing: 12) {
            Button("打开 MOBI/AZW3…") { openFile() }
            Button("上一页") { move(to: currentIndex - 1) }
                .disabled(pages.isEmpty || currentIndex <= 0)
            Button("下一页") { move(to: currentIndex + 1) }
                .disabled(pages.isEmpty || currentIndex >= pages.count - 1)
            Spacer()
            Text(status)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
        }
        .padding(12)
    }

    @ViewBuilder
    private var content: some View {
        if pages.isEmpty {
            VStack(spacing: 10) {
                Image(systemName: "book.closed")
                    .font(.system(size: 48))
                    .foregroundStyle(.secondary)
                Text(status)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            ScrollViewReader { proxy in
                ScrollView(.vertical) {
                    LazyVStack(spacing: 16) {
                        ForEach(pages.indices, id: \.self) { index in
                            pageView(index)
                                .id(index)
                                .onAppear { updateWindow(around: index) }
                        }
                    }
                    .padding(20)
                }
                .onChange(of: currentIndex) { _, newIndex in
                    proxy.scrollTo(newIndex, anchor: .top)
                }
            }
        }
    }

    @ViewBuilder
    private func pageView(_ index: Int) -> some View {
        Group {
            if let image = imageCache[index] {
                Image(image, scale: 1, label: Text("第 \(index + 1) 页"))
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: .infinity)
                    .shadow(radius: 2)
            } else {
                ZStack {
                    RoundedRectangle(cornerRadius: 6).fill(.quaternary)
                    ProgressView("第 \(index + 1) 页")
                        .font(.caption)
                }
                .frame(height: 260)
            }
        }
        .onAppear { load(index) }
    }

    private func openFile() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowedContentTypes = ["com.amazon.mobi", "com.amazon.azw3", "com.amazon.prc"]
            .compactMap { UTType($0) }
        guard panel.runModal() == .OK, let url = panel.url else { return }
        sourceURL = url
        status = "正在解析 \(url.lastPathComponent)…"
        pages = []
        stitchedRanges = []
        imageCache = [:]
        currentIndex = 0
        let config = model.config
        Task.detached(priority: .userInitiated) {
            do {
                let parser = MobiParser(config: config)
                guard let document = try parser.scanDocument(at: url) else {
                    throw AgentError.parse("该文件不是扫描版 MOBI/AZW3，或没有可提取的图片")
                }
                await MainActor.run {
                    pages = document.pages
                    stitchedRanges = document.stitchedRanges
                    status = "\(url.lastPathComponent)：\(pages.count) 页"
                        + (stitchedRanges.isEmpty ? "" : "，\(stitchedRanges.count) 组拼接")
                    if !pages.isEmpty { load(0) }
                }
            } catch {
                await MainActor.run { status = "解析失败：\(error.localizedDescription)" }
            }
        }
    }

    private func move(to index: Int) {
        guard pages.indices.contains(index) else { return }
        currentIndex = index
        updateWindow(around: index)
    }

    /// 只保留当前页 ±1 的 CGImage；其它页立即释放并根据新窗口预取。
    private func updateWindow(around index: Int) {
        guard !pages.isEmpty else { return }
        currentIndex = index
        let lower = max(0, index - 1)
        let upper = min(pages.count - 1, index + 1)
        let keep = Set(lower...upper)
        for key in Array(imageCache.keys) where !keep.contains(key) {
            imageCache.removeValue(forKey: key)
        }
        for candidate in keep { load(candidate) }
    }

    private func load(_ index: Int) {
        guard pages.indices.contains(index), imageCache[index] == nil else { return }
        if let image = pages[index].cgImage() {
            imageCache[index] = image
        }
    }
}
