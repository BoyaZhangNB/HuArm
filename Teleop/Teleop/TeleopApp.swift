//
//  TeleopApp.swift
//  Teleop
//
//  Created by 张博亚 on 2026-08-11.
//

import SwiftUI

@main
struct TeleopApp: App {
    var body: some Scene {
        WindowGroup {
            // Two independent pages, each with its own connection and
            // stream: ContentView teleoperates the arm through teleop.py,
            // BowCommandView hands a running inference.py policy the bow
            // velocity/pressure it should chase. A TabView (rather than a
            // push) keeps both alive while the other is on screen, so
            // switching pages never tears down an active stream.
            TabView {
                ContentView()
                    .tabItem {
                        Label("Arm", systemImage: "hand.draw")
                    }

                BowCommandView()
                    .tabItem {
                        Label("Bow", systemImage: "slider.horizontal.3")
                    }
            }
            .tint(.indigo)
        }
    }
}
