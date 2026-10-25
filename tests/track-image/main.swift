// Checks the iPhone app's path into the shared C++ code on the Mac: loads an image the way
// the app sees camera frames (BGRA), tracks the face and builds the liner.
//
//   swift run lashline-track-image photo.png
//
// Prints the outer eye corners and the liner's vertex count, to compare with
// `build-tests/track-video` on the same frame.

import CoreGraphics
import Foundation
import ImageIO
import LashlineCore

guard CommandLine.arguments.count == 2 else {
    print("usage: lashline-track-image IMAGE")
    exit(2)
}

let models = URL(filePath: #filePath).deletingLastPathComponent().appending(path: "../../data/models").standardized
var error = [CChar](repeating: 0, count: 512)
guard let tracker = lashline_tracker_create(models.appending(path: "face_detector.onnx").path,
                                            models.appending(path: "face_landmarks_detector.onnx").path,
                                            &error, error.count) else {
    print("couldn't load the models: \(String(decoding: error.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self))")
    exit(1)
}
defer { lashline_tracker_destroy(tracker) }

guard let source = CGImageSourceCreateWithURL(URL(filePath: CommandLine.arguments[1]) as CFURL, nil),
      let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
    print("couldn't read \(CommandLine.arguments[1])")
    exit(1)
}

// BGRA, the same byte order as the iPhone camera's frames.
let width = image.width, height = image.height, stride = width * 4
var pixels = [UInt8](repeating: 0, count: stride * height)
let bitmapInfo = CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
pixels.withUnsafeMutableBytes { buffer in
    let context = CGContext(data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8,
                            bytesPerRow: stride, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: bitmapInfo)!
    context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
}

var points = [LashlinePoint](repeating: LashlinePoint(), count: Int(LASHLINE_FACE_POINT_COUNT))
// The tracker refines its face region from the previous frame, so run it twice like a
// second video frame would.
for _ in 0..<2 {
    guard lashline_track_bgra(tracker, pixels, Int32(width), Int32(height), Int32(stride), &points) else {
        print("no face")
        exit(1)
    }
}
print(String(format: "right eye corner (33): %.1f, %.1f", points[33].x, points[33].y))
print(String(format: "left eye corner (263): %.1f, %.1f", points[263].x, points[263].y))

var look = lashline_default_look()
let vertexCount = lashline_liner_triangles(points, &look, nil, 0)
print("liner: \(vertexCount) vertices")
