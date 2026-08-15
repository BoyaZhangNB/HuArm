//
//  TCPClient.swift
//  Teleop
//
//  Minimal newline-delimited JSON sender over TCP, built on the Network
//  framework. Used for control messages (reset/collect) that need reliable
//  delivery, unlike the fire-and-forget position stream sent over UDP.
//
//  TCP is a byte stream with no built-in message boundaries, so each send
//  is terminated with "\n" -- the receiving side should read line-delimited
//  JSON (e.g. Python's `socket.makefile().readline()`).
//

import Foundation
import Network

final class TCPClient {
    private var connection: NWConnection?
    private let queue = DispatchQueue(label: "com.teleop.tcpclient")

    private(set) var isReady = false

    func connect(host: String, port: UInt16) {
        disconnect()

        guard let nwPort = NWEndpoint.Port(rawValue: port) else { return }
        let endpointHost = NWEndpoint.Host(host)
        let params = NWParameters.tcp

        let conn = NWConnection(host: endpointHost, port: nwPort, using: params)
        conn.stateUpdateHandler = { [weak self] state in
            switch state {
            case .ready:
                self?.isReady = true
            case .failed, .cancelled:
                self?.isReady = false
            default:
                break
            }
        }
        conn.start(queue: queue)
        connection = conn
    }

    func send(_ data: Data) {
        guard let connection, isReady else { return }
        var framed = data
        framed.append(0x0A) // "\n"
        connection.send(content: framed, completion: .contentProcessed { _ in })
    }

    func disconnect() {
        connection?.cancel()
        connection = nil
        isReady = false
    }
}
