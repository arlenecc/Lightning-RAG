#include "CPDFShim.h"
#include <string.h>
#include <math.h>
#include <stdbool.h>

static bool dict_name_equals(CGPDFDictionaryRef dict, const char *key, const char *expected) {
    const char *name = NULL;
    return CGPDFDictionaryGetName(dict, key, &name) && name
        && strcmp(name, expected) == 0;
}

static bool filter_contains_dct(CGPDFDictionaryRef dict) {
    if (dict_name_equals(dict, "Filter", "DCTDecode")) return true;
    CGPDFArrayRef array = NULL;
    if (!CGPDFDictionaryGetArray(dict, "Filter", &array) || !array) return false;
    __block bool found = false;
    CGPDFArrayApplyBlock(array, ^bool (size_t index, CGPDFObjectRef value, void *info) {
        const char *name = NULL;
        if (CGPDFObjectGetValue(value, kCGPDFObjectTypeString, &name) && name
            && strcmp(name, "DCTDecode") == 0) {
            found = true;
            return false;
        }
        return true;
    }, NULL);
    return found;
}

typedef struct {
    CGPDFStreamRef streams[8];
    int count;
} StreamCollector;

/// 检查页面资源里是否存在「可作为 JPEG 直提取」的图片流：
/// Subtype=Image、DCTDecode、无 SMask/Mask、8 bit、尺寸达标。
/// JPEG 字节的来源有两种：
/// - `CGPDFStreamCopyData` 报告 JPEGEncoded → CoreGraphics 认可，直接用；
/// - 报告 raw 但字节以 SOI (FFD8) 开头 → DCTDecode 的原始流就是 JPEG 字节
///   流。CoreGraphics 的 JPEG 识别器对带 Photoshop APP13 / 多段 Exif 的
///   书商扫描件可能拒绝（表现为 `CGContextDrawPDFPage` 渲染全白、format
///   降级 raw），而 ImageIO 宽容得多——实测这类文件 sips 能正常解码。
/// 找不到返回 NULL。`largestOnly` 为 true 时在多个候选里挑像素数最大的
/// （用于渲染全白兜底），为 false 时要求全页唯一图片流（正常扫描页）。
static CFDataRef copyEmbeddedJPEG(CGPDFPageRef page, bool largestOnly) {
    if (!page) return NULL;
    if (((CGPDFPageGetRotationAngle(page) % 360) + 360) % 360 != 0) return NULL;

    CGPDFDictionaryRef pageDict = CGPDFPageGetDictionary(page);
    if (!pageDict) return NULL;
    CGPDFDictionaryRef resources = NULL;
    if (!CGPDFDictionaryGetDictionary(pageDict, "Resources", &resources) || !resources) return NULL;
    CGPDFDictionaryRef xobjects = NULL;
    if (!CGPDFDictionaryGetDictionary(resources, "XObject", &xobjects) || !xobjects) return NULL;

    // 收集本页全部流对象（防御异常文件上限 8）。
    StreamCollector collector;
    memset(&collector, 0, sizeof(collector));
    CGPDFDictionaryApplyBlock(xobjects,
        ^bool (const char *key, CGPDFObjectRef value, void *info) {
        StreamCollector *c = (StreamCollector *)info;
        CGPDFStreamRef stream = NULL;
        if (CGPDFObjectGetValue(value, kCGPDFObjectTypeStream, &stream) && stream
            && c->count < 8) {
            c->streams[c->count++] = stream;
        }
        return true;
    }, &collector);
    if (collector.count < 1 || collector.count > 8) return NULL;
    if (!largestOnly && collector.count != 1) return NULL;

    CGRect box = CGPDFPageGetBoxRect(page, kCGPDFCropBox);
    double pageAR = (box.size.width > 0 && box.size.height > 0)
        ? (double)box.size.width / (double)box.size.height : 0;

    CFDataRef best = NULL;
    double bestPixels = 0;
    int candidates = 0;
    for (int i = 0; i < collector.count; i++) {
        CGPDFStreamRef stream = collector.streams[i];
        CGPDFDictionaryRef streamDict = CGPDFStreamGetDictionary(stream);
        if (!streamDict) continue;
        if (!dict_name_equals(streamDict, "Subtype", "Image")) continue;
        if (!filter_contains_dct(streamDict)) continue;

        CGPDFObjectRef object = NULL;
        if (CGPDFDictionaryGetObject(streamDict, "SMask", &object)) continue;
        if (CGPDFDictionaryGetObject(streamDict, "Mask", &object)) continue;

        CGPDFInteger width = 0, height = 0, bitsPerComponent = 0;
        if (!CGPDFDictionaryGetInteger(streamDict, "Width", &width)
            || !CGPDFDictionaryGetInteger(streamDict, "Height", &height)
            || width < 64 || height < 64
            || !CGPDFDictionaryGetInteger(streamDict, "BitsPerComponent", &bitsPerComponent)
            || bitsPerComponent != 8) continue;

        // 与 Swift 侧 PDFPageRenderer.maxEmbeddedPixelCount 保持一致：
        // 超高像素图片放弃直提取，交给有界栅格化（目标长边 1408/1500）。
        // 否则 ImageIO 解码器仍得面对数亿像素，单页就可能要几 GB 内存。
        if (width > 20000 || height > 20000
            || (double)width * (double)height > 200000000.0) continue;

        if (pageAR > 0) {
            double imageAR = (double)width / (double)height;
            if (fabs(pageAR - imageAR) / pageAR > 0.25) continue;
        }

        CGPDFDataFormat format = CGPDFDataFormatRaw;
        CFDataRef data = CGPDFStreamCopyData(stream, &format);
        if (!data) continue;
        bool isJPEG = false;
        if (format == CGPDFDataFormatJPEGEncoded) {
            isJPEG = true;
        } else if (format == CGPDFDataFormatRaw) {
            const UInt8 *bytes = CFDataGetBytePtr(data);
            CFIndex length = CFDataGetLength(data);
            isJPEG = length > 4 && bytes[0] == 0xFF && bytes[1] == 0xD8;
        }
        if (!isJPEG) {
            CFRelease(data);
            continue;
        }
        candidates++;
        if (!largestOnly) return data;
        double pixels = (double)width * (double)height;
        if (pixels > bestPixels) {
            if (best) CFRelease(best);
            best = data;
            bestPixels = pixels;
        } else {
            CFRelease(data);
        }
    }
    if (!largestOnly && candidates == 0) return NULL;
    return best;
}

CFDataRef CPDFShimCopyEmbeddedPageJPEG(CGPDFPageRef page) {
    return copyEmbeddedJPEG(page, false);
}

CFDataRef CPDFShimCopyLargestEmbeddedJPEG(CGPDFPageRef page) {
    return copyEmbeddedJPEG(page, true);
}
