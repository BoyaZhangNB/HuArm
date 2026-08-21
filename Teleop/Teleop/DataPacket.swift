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
//  `CommandPacket` is the bow target the operator dials in on the Bow
//  Command page; like position it is streamed continuously and is
//  loss-tolerant (the robot holds the last value it received), so it also
//  goes out over UDP -- but to its own port, since it is consumed by
//  inference.py (a running policy) rather than teleop.py.
//
//  PositionPacket, one JSON object per UDP datagram:
//  {"x": <float>, "y": <float>, "z": <float>, "pitch": <float>}
//
//  ControlPacket, one JSON object per newline-terminated TCP send:
//  {"reset": <bool>, "collect": <bool>}
//
//  CommandPacket, one JSON object per UDP datagram:
//  {"velocity": <double>, "pressure": <double>}
//
//  velocity is the desired signed lateral bow speed in m/s (its sign is the
//  stroke direction along the erhu's own left/right axis) and pressure is
//  the desired bow-hair/A-string contact force in newtons. Both fields are
//  always sent: inference.py drops a datagram missing either rather than
//  half-applying it.
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

struct CommandPacket: Codable {
    let velocity: Double
    let pressure: Double

    func encoded() -> Data? {
        try? JSONEncoder().encode(self)
    }
}

/// Defaults shared by both pages, so the robot's address and the port each
/// receiver listens on are stated once.
///
/// The three ports match the Python side's own defaults: teleop.py binds
/// 5005 (position, UDP) and 5006 (control, TCP); inference.py binds 5007
/// for bow commands (its --command-port).
enum RobotDefaults {
    static let host = "192.168.1.149"
    static let positionPort = "5005"
    static let controlPort = "5006"
    static let commandPort = "5007"
}

/// Limits of the bow-command sliders. These mirror ErhuEnv's traj_v_limit /
/// traj_p_max defaults -- the span of the scripted reference stroke the
/// policy was trained against. inference.py clips anything outside this
/// range back into it (unless run with --no-clip-command), so sending
/// further would not buy any extra bow speed or force, only an
/// off-distribution target the policy never saw.
enum BowCommandLimits {
    static let velocityLimit: Double = 0.1   // m/s, symmetric
    static let pressureMax: Double = 3.0     // N
}
