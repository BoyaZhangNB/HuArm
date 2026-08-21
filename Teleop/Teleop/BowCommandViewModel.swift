//
//  BowCommandViewModel.swift
//  Teleop
//
//  Drives the Bow Command page: two sliders (desired bow velocity and
//  pressure) streamed to a running inference.py as `CommandPacket`s.
//
//  This is a separate send path from TeleopViewModel's, not a second use of
//  it: the arm page talks to teleop.py (which drives the arm from the
//  phone's AR pose), while these two numbers are consumed by inference.py,
//  where a trained policy -- not the operator -- moves the arm and the
//  operator only sets the target it chases. The two Python programs bind
//  different ports and are usually run one at a time, so the pages keep
//  independent connections and stream state.
//
//  Like position, commands are streamed continuously at a fixed rate rather
//  than sent on slider change: UDP is lossy, and inference.py holds the last
//  value it received (zero-order hold), so re-sending is what makes a
//  dropped datagram self-correcting instead of leaving the policy chasing a
//  stale target.
//

import Combine
import Foundation

@MainActor
final class BowCommandViewModel: ObservableObject {
    @Published var host: String = RobotDefaults.host
    @Published var port: String = RobotDefaults.commandPort
    @Published var isStreaming = false
    @Published var sendRateHz: Double = 30

    /// Desired signed lateral bow velocity, m/s. Sign is stroke direction.
    @Published var velocity: Double = 0
    /// Desired bow-hair/A-string contact force, newtons.
    @Published var pressure: Double = 0

    @Published private(set) var packetsSent = 0
    @Published private(set) var lastSentAt: Date?

    private let udpClient = UDPClient()
    private var timerCancellable: AnyCancellable?

    var canStart: Bool {
        UInt16(port) != nil && !host.trimmingCharacters(in: .whitespaces).isEmpty
    }

    func startStreaming() {
        guard let portNum = UInt16(port) else { return }
        udpClient.connect(host: host, port: portNum)
        isStreaming = true

        timerCancellable = Timer.publish(every: 1.0 / sendRateHz, on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in self?.sendCommandPacket() }
    }

    /// Stops the stream, but sends one last zero-velocity packet first:
    /// inference.py holds whatever it last received, so simply going quiet
    /// mid-stroke would leave the policy driving the bow at the last
    /// commanded speed indefinitely. Pressure is left as-is -- resting the
    /// bow on the string is harmless, while yanking it off is not what the
    /// operator asked for.
    func stopStreaming() {
        if isStreaming {
            velocity = 0
            sendCommandPacket()
        }
        timerCancellable?.cancel()
        timerCancellable = nil
        udpClient.disconnect()
        isStreaming = false
    }

    /// Snaps the bow to a standstill -- a slider is a clumsy way to hit
    /// exactly zero, and zero is the one value the operator reaches for most
    /// (hold the current pressure, stop moving).
    func stopBow() {
        velocity = 0
    }

    /// Lifts the bow off the string entirely: no motion, no force.
    func releaseBow() {
        velocity = 0
        pressure = 0
    }

    private func sendCommandPacket() {
        guard isStreaming else { return }

        let packet = CommandPacket(velocity: velocity, pressure: pressure)
        guard let data = packet.encoded() else { return }
        udpClient.send(data)
        packetsSent += 1
        lastSentAt = Date()
    }
}
