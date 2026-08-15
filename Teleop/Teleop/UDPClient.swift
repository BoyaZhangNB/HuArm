//
//  UDPClient.swift
//  Teleop
//
//  Minimal fire-and-forget UDP sender built on the Network framework.
//

import Foundation
import Network

final class UDPClient {
    private var connection: NWConnection?
    private let queue = DispatchQueue(label: "com.teleop.udpclient")

    private(set) var isReady = false

    func connect(host: String, port: UInt16) {
        disconnect()

        guard let nwPort = NWEndpoint.Port(rawValue: port) else { return }
        let endpointHost = NWEndpoint.Host(host)
        let params = NWParameters.udp
        params.allowLocalEndpointReuse = true

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
        guard let connection else { return }
        connection.send(content: data, completion: .contentProcessed { _ in
            // Fire-and-forget: UDP has no delivery guarantee, so we don't retry on error.
        })
    }

    func disconnect() {
        connection?.cancel()
        connection = nil
        isReady = false
    }
}
