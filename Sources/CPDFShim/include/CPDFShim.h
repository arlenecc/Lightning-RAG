#ifndef CPDFSHIM_H
#define CPDFSHIM_H
#include <CoreGraphics/CoreGraphics.h>

/// 提取扫描页唯一的 DCTDecode（JPEG）整页图像，跳过栅格化。
///
/// 全部条件满足才返回（否则 NULL，调用方回退 CoreGraphics 渲染）：
/// - 页面旋转角为 0（/Rotate 需要重排位图，直接走渲染路径更稳）；
/// - 页面 XObject 中图像流恰好 1 张（多图页面不启发式拼合）；
/// - Subtype == Image，Filter 含 DCTDecode（单名或数组）；
/// - 无 SMask / Mask（带蒙版需与页面合成）；
/// - BitsPerComponent == 8，宽高 ≥ 64；
/// - 图像长宽比与页面裁剪框差 ≤ 25%（过滤插图误判为整页图）。
///
/// @return 满足条件时返回 retained CFDataRef（JPEG 字节流，调用方负责
///         CFRelease），否则 NULL。
CFDataRef CPDFShimCopyEmbeddedPageJPEG(CGPDFPageRef page);

/// 直提取兜底：页面栅格化结果全白（CoreGraphics 拒绝解码某些书商扫描件
/// 的 JPEG，如带 Photoshop APP13 段的文件）但页面确实有 DCTDecode 图片时，
/// 挑像素数最大的合格图片流直接返回其 JPEG 字节。接受 CoreGraphics 报告
/// raw 但以 SOI (FFD8) 开头的数据（ImageIO 比 CGPDF 的 JPEG 识别器宽容）。
/// 同样接受 raw 但以 SOI (FFD8) 开头的数据。
///
/// @return retained CFDataRef（JPEG 字节流，调用方负责 CFRelease），否则 NULL。
CFDataRef CPDFShimCopyLargestEmbeddedJPEG(CGPDFPageRef page);

#endif /* CPDFSHIM_H */
