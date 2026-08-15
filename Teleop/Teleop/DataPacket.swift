//
//  DataPacket.swift
//  Teleop
//
//  Defines PositionPacket and ControlPacket, the two wire formats sent to
//  the robot (see below).
//
//  Wire formats sent to the robot. Position/pitch is high-rate and
//  loss-tolerant, so it goes out over UDP as `PositionPacket`. reset/collect
//  are state changes that must arrive, so they go out over TCP as
//  `ControlPacket` (see TCPClient for the newline-delimited framing).
//
//  PositionPacket, one JSON object per UDP datagram:
//  {"x": <float>, "y": <float>, "z": <float>, "pitch": <float>}
//
//  ControlPacket, one JSON object per newline-terminated TCP send:
//  {"reset": <bool>, "collect": <bool>}
//
//  pitch is radians, set directly by the operator via the pitch slider (not
//  tracked from the phone's orientation), re-homed the same way as x/y/z
//  whenever the operator resets the origin.
//
//  On the receiving (Python) side, `json.loads(datagram)` turns this into a
//  dict with True/False for the booleans, matching the requested schema.
//

import Foundation

struct PositionPacket: Codable {
    let x: Double
    let y: Double
    let z: Double
    let pitch: Double

    func encoded() -> Data? {
        try? JSONEncoder().encode(self)
    }
}

struct ControlPacket: Codable {
    let reset: Bool
    let collect: Bool

    func encoded() -> Data? {
        try? JSONEncoder().encode(self)
    }
}
