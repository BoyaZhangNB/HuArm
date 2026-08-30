//
//  TeleopViewModel.swift
//  Teleop
//
//  Ties together AR position tracking, the stiffness slider, the
//  reset/collect button state, and the network send paths.
//  Position/stiffness packets go out over UDP at a fixed rate off the AR
//  frame callback (which runs at ~60 Hz) so the network isn't flooded.
//  reset/collect are one-off state changes, so they're sent immediately over
//  TCP for reliable delivery instead of riding the UDP loop.
//

import Combine
import Foundation

@MainActor
final class TeleopViewModel: ObservableObject {
    @Published var host: String = RobotDefaults.host
    @Published var port: String = RobotDefaults.positionPort
    @Published var tcpPort: String = RobotDefaults.controlPort
    @Published var isStreaming = false
    @Published var isCollecting = false
    @Published var lastResetAt: Date?
    @Published var sendRateHz: Double = 30
    /// bow_frog_hinge friction-clamp target, a [0, 1] fraction (0 = loosest,
    /// 1 = tightest) set directly by the operator via the slider -- see
    /// PositionPacket's `stiffness` field.
    @Published var stiffness: Double = 0

    let arManager = ARPositionManager()
    private let udpClient = UDPClient()
    private let tcpClient = TCPClient()
    private var timerCancellable: AnyCancellable?
    private var arManagerCancellable: AnyCancellable?

    init() {
        // arManager is a plain property, not @Published, so its own change
        // notifications (position updates on every AR frame) wouldn't
        // otherwise reach views observing this view model. Forward them so
        // the UI stays live.
        arManagerCancellable = arManager.objectWillChange.sink { [weak self] _ in
            self?.objectWillChange.send()
        }
    }

    var canStart: Bool {
        UInt16(port) != nil && UInt16(tcpPort) != nil
            && !host.trimmingCharacters(in: .whitespaces).isEmpty
    }

    func startStreaming() {
        guard let portNum = UInt16(port) else { return }
        guard let tcpPortNum = UInt16(tcpPort) else { return }
        arManager.start()
        udpClient.connect(host: host, port: portNum)
        tcpClient.connect(host: host, port: tcpPortNum)
        isStreaming = true

        timerCancellable = Timer.publish(every: 1.0 / sendRateHz, on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in self?.sendPositionPacket() }
    }

    func stopStreaming() {
        timerCancellable?.cancel()
        timerCancellable = nil
        arManager.stop()
        udpClient.disconnect()
        tcpClient.disconnect()
        isStreaming = false
    }

    /// Zeroes x/y/z, re-homes future position readings to the current pose,
    /// and sends a single reset=true control packet over TCP.
    func triggerReset() {
        arManager.resetOffset()
        lastResetAt = Date()
        sendControlPacket(reset: true)
    }

    func toggleCollect() {
        isCollecting.toggle()
        sendControlPacket(reset: false)
    }

    private func sendPositionPacket() {
        guard isStreaming else { return }

        let pos = arManager.offsetPosition
        let packet = PositionPacket(
            x: pos.x,
            y: pos.y,
            z: pos.z,
            stiffness: stiffness
        )
        guard let data = packet.encoded() else { return }
        udpClient.send(data)
    }

    private func sendControlPacket(reset: Bool) {
        guard isStreaming else { return }

        let packet = ControlPacket(reset: reset, collect: isCollecting)
        guard let data = packet.encoded() else { return }
        tcpClient.send(data)
    }
}
