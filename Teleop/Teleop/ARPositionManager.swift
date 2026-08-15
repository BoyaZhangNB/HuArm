//
//  ARPositionManager.swift
//  Teleop
//
//  Tracks the iPhone's position in 3D space using ARKit's visual-inertial
//  odometry (world tracking). This is the standard way to get a metric x/y/z
//  position from an iPhone -- CoreMotion alone only gives acceleration /
//  orientation, not position.
//
//  ARKit's camera transform reports position in meters, in a world frame
//  fixed at the point tracking started. `resetOffset()` re-zeros that frame
//  to the phone's current location, so the app can be "re-homed" whenever
//  the operator wants (x,y,z) == (0,0,0) at the current pose.
//

import ARKit
import Combine

@MainActor
final class ARPositionManager: NSObject, ObservableObject {
    @Published private(set) var rawPosition = SIMD3<Double>(repeating: 0)
    @Published private(set) var offsetPosition = SIMD3<Double>(repeating: 0)
    @Published private(set) var trackingStateDescription = "Not started"
    @Published private(set) var isRunning = false

    private let session = ARSession()
    private var offset = SIMD3<Double>(repeating: 0)

    override init() {
        super.init()
        session.delegate = self
    }

    var isSupported: Bool {
        ARWorldTrackingConfiguration.isSupported
    }

    func start() {
        guard isSupported else {
            trackingStateDescription = "ARKit not supported on this device"
            return
        }
        let config = ARWorldTrackingConfiguration()
        config.worldAlignment = .gravity
        config.planeDetection = []
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
        isRunning = true
    }

    func stop() {
        session.pause()
        isRunning = false
        trackingStateDescription = "Stopped"
    }

    /// Re-zeroes (x, y, z) to the phone's current position.
    func resetOffset() {
        offset = rawPosition
        offsetPosition = .zero
    }
}

extension ARPositionManager: ARSessionDelegate {
    nonisolated func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let t = frame.camera.transform.columns.3
        let raw = SIMD3<Double>(Double(t.x), Double(t.y), Double(t.z))
        let tracking = frame.camera.trackingState

        Task { @MainActor in
            self.rawPosition = raw
            self.offsetPosition = raw - self.offset
            switch tracking {
            case .normal:
                self.trackingStateDescription = "Normal"
            case .notAvailable:
                self.trackingStateDescription = "Not available"
            case .limited(.excessiveMotion):
                self.trackingStateDescription = "Limited (moving too fast)"
            case .limited(.insufficientFeatures):
                self.trackingStateDescription = "Limited (low detail / low light)"
            case .limited(.initializing):
                self.trackingStateDescription = "Initializing…"
            case .limited(.relocalizing):
                self.trackingStateDescription = "Relocalizing…"
            case .limited:
                self.trackingStateDescription = "Limited"
            }
        }
    }

    nonisolated func session(_ session: ARSession, didFailWithError error: Error) {
        Task { @MainActor in
            self.trackingStateDescription = "Error: \(error.localizedDescription)"
            self.isRunning = false
        }
    }
}
