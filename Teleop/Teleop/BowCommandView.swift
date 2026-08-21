//
//  BowCommandView.swift
//  Teleop
//
//  The Bow Command page: sets the desired bow velocity and pressure a
//  trained policy should chase, and streams them to inference.py over UDP.
//  See BowCommandViewModel for why this is its own connection rather than
//  part of the arm teleoperation page.
//

import SwiftUI

struct BowCommandView: View {
    @StateObject private var viewModel = BowCommandViewModel()

    var body: some View {
        NavigationStack {
            Form {
                Section("Policy Connection") {
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

                Section("Velocity") {
                    LabeledContent("Desired") {
                        Text(formatVelocity(viewModel.velocity))
                            .monospacedDigit()
                    }
                    Slider(
                        value: $viewModel.velocity,
                        in: -BowCommandLimits.velocityLimit ... BowCommandLimits.velocityLimit
                    ) {
                        Text("Velocity")
                    } minimumValueLabel: {
                        Text(formatVelocity(-BowCommandLimits.velocityLimit))
                    } maximumValueLabel: {
                        Text(formatVelocity(BowCommandLimits.velocityLimit))
                    }
                    .monospacedDigit()
                    Text("Signed: negative and positive are the two stroke directions along the erhu's left/right axis; zero holds the bow still.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section("Pressure") {
                    LabeledContent("Desired") {
                        Text(formatPressure(viewModel.pressure))
                            .monospacedDigit()
                    }
                    Slider(
                        value: $viewModel.pressure,
                        in: 0 ... BowCommandLimits.pressureMax
                    ) {
                        Text("Pressure")
                    } minimumValueLabel: {
                        Text(formatPressure(0))
                    } maximumValueLabel: {
                        Text(formatPressure(BowCommandLimits.pressureMax))
                    }
                    .monospacedDigit()
                    Text("Bow-hair force against the A string. Zero lifts the bow off it.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section("Controls") {
                    Button {
                        viewModel.stopBow()
                    } label: {
                        Label("Stop Bow", systemImage: "pause.circle")
                            .frame(maxWidth: .infinity)
                    }
                    .tint(.orange)

                    Button {
                        viewModel.releaseBow()
                    } label: {
                        Label("Release", systemImage: "arrow.up.circle")
                            .frame(maxWidth: .infinity)
                    }
                    .tint(.gray)
                }
                .buttonStyle(.borderedProminent)

                Section("Stream") {
                    LabeledContent("Status") {
                        Text(viewModel.isStreaming ? "Streaming" : "Idle")
                            .foregroundStyle(viewModel.isStreaming ? .green : .secondary)
                    }
                    LabeledContent("Rate") {
                        Text(String(format: "%.0f Hz", viewModel.sendRateHz))
                    }
                    LabeledContent("Packets sent") {
                        Text("\(viewModel.packetsSent)")
                    }
                    if let lastSentAt = viewModel.lastSentAt {
                        LabeledContent("Last sent") {
                            Text(lastSentAt, style: .time)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                .monospacedDigit()
            }
            .navigationTitle("Bow Command")
        }
    }

    private func formatVelocity(_ value: Double) -> String {
        String(format: "% .3f m/s", value)
    }

    private func formatPressure(_ value: Double) -> String {
        String(format: "%.2f N", value)
    }
}

#Preview {
    BowCommandView()
}
