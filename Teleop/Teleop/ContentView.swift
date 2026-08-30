//
//  ContentView.swift
//  Teleop
//

import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = TeleopViewModel()

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: Theme.cardSpacing) {
                    PageHeader(
                        title: "Arm",
                        subtitle: "Position + stiffness \u{2192} teleop.py",
                        statusText: viewModel.isStreaming ? "Streaming" : "Idle",
                        statusColor: viewModel.isStreaming ? .green : .secondary
                    )

                    Card(title: "Robot Connection", systemImage: "network") {
                        VStack(spacing: 10) {
                            TextField("IP address", text: $viewModel.host)
                                .keyboardType(.decimalPad)
                                .autocorrectionDisabled()
                                .disabled(viewModel.isStreaming)
                                .fieldStyle()

                            HStack(spacing: 10) {
                                TextField("UDP port", text: $viewModel.port)
                                    .keyboardType(.numberPad)
                                    .disabled(viewModel.isStreaming)
                                    .fieldStyle()

                                TextField("TCP port", text: $viewModel.tcpPort)
                                    .keyboardType(.numberPad)
                                    .disabled(viewModel.isStreaming)
                                    .fieldStyle()
                            }

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

                    Card(title: "Live Position", systemImage: "location") {
                        VStack(spacing: 8) {
                            MetricRow(label: "Tracking", value: viewModel.arManager.trackingStateDescription)
                            Divider()
                            MetricRow(label: "X", value: format(viewModel.arManager.offsetPosition.x))
                            MetricRow(label: "Y", value: format(viewModel.arManager.offsetPosition.y))
                            MetricRow(label: "Z", value: format(viewModel.arManager.offsetPosition.z))
                            if let lastReset = viewModel.lastResetAt {
                                Divider()
                                MetricRow(
                                    label: "Last reset",
                                    value: lastReset.formatted(date: .omitted, time: .standard),
                                    valueColor: .secondary
                                )
                            }
                        }
                    }

                    Card(title: "Stiffness", systemImage: "dial.medium") {
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Text("Target")
                                    .foregroundStyle(.secondary)
                                Spacer()
                                Text(formatFraction(viewModel.stiffness))
                                    .monospacedDigit()
                                    .font(.title3.weight(.semibold))
                            }
                            Slider(value: $viewModel.stiffness, in: 0 ... 1)
                            Text("bow_frog_hinge friction-clamp target: 0 is loosest (passive), 1 is tightest.")
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        }
                    }

                    Card(title: "Controls", systemImage: "hand.tap") {
                        VStack(spacing: 10) {
                            Button {
                                viewModel.triggerReset()
                            } label: {
                                Label("Reset Origin", systemImage: "arrow.counterclockwise")
                                    .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(.indigo)
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
                            .buttonStyle(.borderedProminent)
                            .tint(viewModel.isCollecting ? .green : .gray)
                            .disabled(!viewModel.isStreaming)
                        }
                    }
                }
                .padding()
            }
            .background(Color(.systemGroupedBackground))
            .navigationBarHidden(true)
        }
    }

    private func format(_ value: Double) -> String {
        String(format: "% .3f m", value)
    }

    private func formatFraction(_ value: Double) -> String {
        String(format: "%.2f", value)
    }
}

#Preview {
    ContentView()
}
