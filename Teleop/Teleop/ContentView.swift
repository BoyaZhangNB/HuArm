//
//  ContentView.swift
//  Teleop
//

import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = TeleopViewModel()

    var body: some View {
        NavigationStack {
            Form {
                Section("Robot Connection") {
                    TextField("IP address", text: $viewModel.host)
                        .keyboardType(.decimalPad)
                        .autocorrectionDisabled()
                        .disabled(viewModel.isStreaming)

                    TextField("UDP port", text: $viewModel.port)
                        .keyboardType(.numberPad)
                        .disabled(viewModel.isStreaming)

                    Button {
                        viewModel.isStreaming ? viewModel.stopStreaming() : viewModel.startStreaming()
                    } label: {
                        Text(viewModel.isStreaming ? "Stop Streaming" : "Start Streaming")
                            .frame(maxWidth: .infinity)
                    }
                    .disabled(!viewModel.isStreaming && !viewModel.canStart)
                    .tint(viewModel.isStreaming ? .red : .accentColor)
                }

                Section("Live Position") {
                    LabeledContent("Tracking") {
                        Text(viewModel.arManager.trackingStateDescription)
                            .foregroundStyle(.secondary)
                    }
                    LabeledContent("X") {
                        Text(format(viewModel.arManager.offsetPosition.x))
                    }
                    LabeledContent("Y") {
                        Text(format(viewModel.arManager.offsetPosition.y))
                    }
                    LabeledContent("Z") {
                        Text(format(viewModel.arManager.offsetPosition.z))
                    }
                    if let lastReset = viewModel.lastResetAt {
                        LabeledContent("Last reset") {
                            Text(lastReset, style: .time)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                .monospacedDigit()

                Section("Pitch") {
                    LabeledContent("Pitch") {
                        Text(formatDegrees(viewModel.pitch))
                            .monospacedDigit()
                    }
                    Slider(value: $viewModel.pitch, in: -.pi / 2 ... .pi / 2)
                }

                Section("Controls") {
                    Button {
                        viewModel.triggerReset()
                    } label: {
                        Label("Reset Origin", systemImage: "arrow.counterclockwise")
                            .frame(maxWidth: .infinity)
                    }
                    .disabled(!viewModel.isStreaming)

                    Button {
                        viewModel.toggleCollect()
                    } label: {
                        Label(
                            viewModel.isCollecting ? "Collecting: ON" : "Collecting: OFF",
                            systemImage: viewModel.isCollecting ? "record.circle.fill" : "record.circle"
                        )
                        .frame(maxWidth: .infinity)
                    }
                    .tint(viewModel.isCollecting ? .green : .gray)
                    .disabled(!viewModel.isStreaming)
                }
                .buttonStyle(.borderedProminent)
            }
            .navigationTitle("Teleop Arm")
        }
    }

    private func format(_ value: Double) -> String {
        String(format: "% .3f m", value)
    }

    private func formatDegrees(_ radians: Double) -> String {
        String(format: "% .1f°", radians * 180 / .pi)
    }
}

#Preview {
    ContentView()
}
