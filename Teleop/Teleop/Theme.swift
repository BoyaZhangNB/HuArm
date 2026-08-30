//
//  Theme.swift
//  Teleop
//
//  Small shared design system for both pages: a card container, a section
//  header, and a status pill. Kept deliberately minimal -- system colors and
//  materials throughout, so it adapts to light/dark for free and never
//  drifts from what native controls (sliders, buttons) already look like.
//

import SwiftUI

enum Theme {
    static let cardSpacing: CGFloat = 16
    static let cardPadding: CGFloat = 16
    static let cardRadius: CGFloat = 18
}

/// A rounded, lightly-elevated container each page's sections sit in,
/// replacing Form's plain grouped rows.
struct Card<Content: View>: View {
    let title: String
    let systemImage: String
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Label(title, systemImage: systemImage)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.secondary)
            content
        }
        .padding(Theme.cardPadding)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: Theme.cardRadius, style: .continuous))
    }
}

/// A colored dot + label, e.g. "Streaming" / "Idle" -- the at-a-glance
/// readout every page's header leads with.
struct StatusPill: View {
    let text: String
    let color: Color

    var body: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(color)
                .frame(width: 8, height: 8)
            Text(text)
                .font(.footnote.weight(.medium))
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(color.opacity(0.15), in: Capsule())
        .foregroundStyle(color)
    }
}

/// Page-top title + status pill, shared by both tabs so they read as one app.
struct PageHeader: View {
    let title: String
    let subtitle: String
    let statusText: String
    let statusColor: Color

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.largeTitle.weight(.bold))
                Text(subtitle)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            StatusPill(text: statusText, color: statusColor)
        }
    }
}

/// A compact "label / value" readout for live telemetry rows, in place of
/// LabeledContent's default styling.
struct MetricRow: View {
    let label: String
    let value: String
    var valueColor: Color = .primary

    var body: some View {
        HStack {
            Text(label)
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .foregroundStyle(valueColor)
                .monospacedDigit()
        }
        .font(.callout)
    }
}

extension View {
    /// Rounded, bordered text field styling shared by both connection cards.
    func fieldStyle() -> some View {
        self
            .textFieldStyle(.roundedBorder)
    }
}
