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
            ScrollView {
                VStack(spacing: Theme.cardSpacing) {
                    PageHeader(
                        title: "Bow",
                        subtitle: "Velocity + pressure \u{2192} inference.py",
                        statusText: viewModel.isStreaming ? "Streaming" : "Idle",
                        statusColor: viewModel.isStreaming ? .green : .secondary
                    )

                    Card(title: "Policy Connection", systemImage: "network") {
                        VStack(spacing: 10) {
                            TextField("IP address", text: $viewModel.host)
                                .keyboardType(.decimalPad)
                                .autocorrectionDisabled()
                                .disabled(viewModel.isStreaming)
                                .fieldStyle()

                            TextField("UDP port", text: $viewModel.port)
                                .keyboardType(.numberPad)
                                .disabled(viewModel.isStreaming)
                                .fieldStyle()

                            Button {
                                viewModel.isStreaming ? viewModel.stopStreaming() : viewModel.startStreaming()
                            } label: {
                                Text(viewModel.isStreaming ? "Stop Streaming" : "Start Streaming")
                                    .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.borderedProminent)
                            .disabled(!viewModel.isStreaming && !viewModel.canStart)
                            .tint(viewModel.isStreaming ? .red : .accentColor)
                        }
                    }

                    Card(title: "Velocity", systemImage: "arrow.left.arrow.right") {
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Text("Desired")
                                    .foregroundStyle(.secondary)
                                Spacer()
                                Text(formatVelocity(viewModel.velocity))
                                    .monospacedDigit()
                                    .font(.title3.weight(.semibold))
                            }
                            Slider(
                                value: $viewModel.velocity,
                                in: -BowCommandLimits.velocityLimit ... BowCommandLimits.velocityLimit
                            ) {
                                Text("Velocity")
                            } minimumValueLabel: {
                                Text(formatVelocity(-BowCommandLimits.velocityLimit))
                                    .font(.caption2)
                            } maximumValueLabel: {
                                Text(formatVelocity(BowCommandLimits.velocityLimit))
                                    .font(.caption2)
                            }
                            .monospacedDigit()
                            Text("Signed: negative and positive are the two stroke directions along the erhu's left/right axis; zero holds the bow still.")
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        }
                    }

                    Card(title: "Pressure", systemImage: "gauge.with.dots.needle.50percent") {
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Text("Desired")
                                    .foregroundStyle(.secondary)
                                Spacer()
                                Text(formatPressure(viewModel.pressure))
                                    .monospacedDigit()
                                    .font(.title3.weight(.semibold))
                            }
                            Slider(
                                value: $viewModel.pressure,
                                in: 0 ... BowCommandLimits.pressureMax
                            ) {
                                Text("Pressure")
                            } minimumValueLabel: {
                                Text(formatPressure(0))
                                    .font(.caption2)
                            } maximumValueLabel: {
                                Text(formatPressure(BowCommandLimits.pressureMax))
                                    .font(.caption2)
                            }
                            .monospacedDigit()
                            Text("Bow-hair force against the A string. Zero lifts the bow off it.")
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        }
                    }

                    Card(title: "Controls", systemImage: "hand.tap") {
                        VStack(spacing: 10) {
                            Button {
                                viewModel.stopBow()
                            } label: {
                                Label("Stop Bow", systemImage: "pause.circle")
                                    .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(.orange)

                            Button {
                                viewModel.releaseBow()
                            } label: {
                                Label("Release", systemImage: "arrow.up.circle")
                                    .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(.gray)
                        }
                    }

                    Card(title: "Stream", systemImage: "dot.radiowaves.up.forward") {
                        VStack(spacing: 8) {
                            MetricRow(
                                label: "Status",
                                value: viewModel.isStreaming ? "Streaming" : "Idle",
                                valueColor: viewModel.isStreaming ? .green : .secondary
                            )
                            MetricRow(label: "Rate", value: String(format: "%.0f Hz", viewModel.sendRateHz))
                            MetricRow(label: "Packets sent", value: "\(viewModel.packetsSent)")
                            if let lastSentAt = viewModel.lastSentAt {
                                MetricRow(
                                    label: "Last sent",
                                    value: lastSentAt.formatted(date: .omitted, time: .standard),
                                    valueColor: .secondary
                                )
                            }
                        }
                    }
                }
                .padding()
            }
            .background(Color(.systemGroupedBackground))
            .navigationBarHidden(true)
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
