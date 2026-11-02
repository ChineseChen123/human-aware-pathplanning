"""Pygame-ce replay viewer: smooth playback of immutable discrete traces."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from .replays import ReplayCatalog, validate_replay

BACKGROUND = (239, 243, 247)
PANEL = (255, 255, 255)
INK = (32, 48, 65)
MUTED = (106, 123, 138)
GRID_LINE = (223, 231, 237)
WALL = (62, 79, 97)
BLUE = (47, 102, 224)
GREEN = (27, 139, 112)
HUMAN_COLORS = [
    (215, 137, 49),
    (170, 95, 177),
    (32, 153, 143),
    (218, 100, 100),
    (88, 133, 184),
    (139, 155, 49),
]


class ReplayViewer:
    def __init__(
        self,
        replay: dict,
        size: tuple[int, int] = (1260, 800),
        *,
        catalog: ReplayCatalog | None = None,
        replay_index: int = 0,
    ) -> None:
        validate_replay(replay)
        self.catalog = catalog
        self.replay_index = replay_index
        self.content_offset = 104 if catalog is not None and catalog.is_batch else 0
        self.dropdown: str | None = None
        self.menu_scroll = 0
        self.menu_boxes: list[tuple[str, pygame.Rect]] = []
        self.notice = ""
        self.window_size: tuple[int, int] | None = None
        self.show_results = False
        self.explorer = None
        self.replay = replay
        self.scenario = replay["scenario"]
        self.metrics = replay["metrics"]
        self.frames = replay["frames"]
        self.baseline = replay["baseline_frames"]
        self.rows = self.scenario["grid"]["rows"]
        self.specs = {a["id"]: a for a in [self.scenario["robot"], *self.scenario["pedestrians"]]}
        self.robot_id = self.scenario["robot"]["id"]
        self.human_ids = sorted(p["id"] for p in self.scenario["pedestrians"])
        self.colors = {
            ident: HUMAN_COLORS[i % len(HUMAN_COLORS)] for i, ident in enumerate(self.human_ids)
        }
        self.width, self.height = size
        self.sidebar_x = self.width - 316
        self.cell = min(
            43,
            (self.sidebar_x - 90) // len(self.rows[0]),
            (self.height - 223 - self.content_offset) // len(self.rows),
        )
        self.cell = max(1, self.cell)
        self.map_x = 44 + (self.sidebar_x - 76 - len(self.rows[0]) * self.cell) // 2
        self.map_y = (
            124
            + self.content_offset
            + (self.height - 240 - self.content_offset - len(self.rows) * self.cell) // 2
        )
        self.playhead = 0.0
        self.playing = True
        self.speed = 1.0
        self.show_halos = True
        self.show_baseline = False
        self.show_path = True
        self.buttons: dict[str, pygame.Rect] = {}
        self.fonts: dict[tuple[int, bool], pygame.font.Font] = {}

    @classmethod
    def from_path(cls, path: str | Path) -> ReplayViewer:
        catalog = ReplayCatalog.open(path)
        return cls(catalog.load(0), size=(1260, 904 if catalog.is_batch else 800), catalog=catalog)

    def _replace_view(self, replacement: ReplayViewer) -> None:
        for preference in (
            "speed",
            "show_halos",
            "show_baseline",
            "show_path",
            "playing",
            "fonts",
            "window_size",
        ):
            setattr(replacement, preference, getattr(self, preference))
        self.__dict__.update(replacement.__dict__)
        if pygame.display.get_init() and pygame.display.get_surface() is not None:
            self._update_caption()

    def switch_replay(self, index: int) -> None:
        if self.catalog is None:
            return
        index %= len(self.catalog.entries)
        try:
            replacement = ReplayViewer(
                self.catalog.load(index),
                (self.width, self.height),
                catalog=self.catalog,
                replay_index=index,
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.notice = f"Could not open replay: {exc}"
            self.dropdown = None
            return
        self._replace_view(replacement)

    def open_path(self, path: str | Path) -> None:
        try:
            replacement = ReplayViewer.from_path(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.notice = f"Could not open results: {exc}"
            return
        self._replace_view(replacement)

    def _choose_batch_folder(self) -> None:
        try:
            import tkinter as tk
            from tkinter import filedialog
        except ImportError:
            self.notice = "Folder picker unavailable. Open a batch with the replay command."
            return
        root = None
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            folder = filedialog.askdirectory(
                parent=root,
                title="Open a navigation results batch",
                initialdir=str(self.catalog.root if self.catalog else Path.cwd() / "results"),
            )
        except tk.TclError as exc:
            self.notice = f"Folder picker unavailable: {exc}"
            return
        finally:
            if root is not None:
                root.destroy()
        if folder:
            self.open_path(folder)

    def _update_caption(self) -> None:
        suffix = (
            f" | Replay {self.replay_index + 1}/{len(self.catalog.entries)}" if self.catalog else ""
        )
        pygame.display.set_caption(f"Human-aware navigation | {self.scenario['name']}{suffix}")

    @property
    def last_tick(self) -> int:
        return max(0, len(self.frames) - 1, len(self.baseline) - 1)

    def font(self, size: int, bold: bool = False) -> pygame.font.Font:
        key = (size, bold)
        if key not in self.fonts:
            self.fonts[key] = pygame.font.SysFont("segoeui", size, bold=bold)
        return self.fonts[key]

    def text(
        self,
        surface: pygame.Surface,
        text: str,
        p: tuple[int, int],
        size: int = 16,
        color: tuple = INK,
        bold: bool = False,
    ) -> None:
        surface.blit(self.font(size, bold).render(str(text), True, color), p)

    def center(self, p: tuple | list) -> tuple[int, int]:
        return (
            round(self.map_x + (p[0] + 0.5) * self.cell),
            round(self.map_y + (p[1] + 0.5) * self.cell),
        )

    def interpolated(self, frames: list[dict], at: float) -> dict[str, tuple[float, float]]:
        index = min(int(at), len(frames) - 1)
        fraction = at - int(at)
        current = frames[index]["positions"]
        following = frames[min(index + 1, len(frames) - 1)]
        result = {}
        for ident, p in current.items():
            q = following["positions"].get(ident, p)
            if following["arrivals"].get(ident) == index + 1:
                q = self.specs[ident]["goals"][-1]
            result[ident] = (p[0] + (q[0] - p[0]) * fraction, p[1] + (q[1] - p[1]) * fraction)
        return result

    def arrived_humans(self, at: float | None = None) -> dict[str, tuple[int, int]]:
        """Decorative destination markers, reconstructed from recorded arrivals.

        These positions never enter the simulator, active-agent rendering, or
        personal-space calculations. Deriving them from the selected frame also
        makes rewinding and existing replay files work without extra state.
        """
        at = self.playhead if at is None else at
        frame = self.frames[max(0, min(int(at), len(self.frames) - 1))]
        return {
            ident: tuple(self.specs[ident]["goals"][-1])
            for ident, arrival in frame["arrivals"].items()
            if ident in self.colors and arrival <= at
        }

    def _draw_arrived_humans(self, surface: pygame.Surface) -> None:
        by_cell = {}
        for ident, position in self.arrived_humans().items():
            by_cell.setdefault(position, []).append(ident)
        for position, people in by_cell.items():
            # Desaturate the pedestrian's color, then blend it into the floor.
            original = self.colors[people[0]]
            gray = sum(
                channel * weight for channel, weight in zip(original, (0.2126, 0.7152, 0.0722))
            )
            muted = [0.8 * gray + 0.2 * channel for channel in original]
            color = tuple(
                round(0.3 * channel + 0.7 * floor) for channel, floor in zip(muted, (247, 250, 252))
            )
            center = self.center(position)
            radius = max(3, round(self.cell * 0.29))
            pygame.draw.circle(surface, color, center, radius)
            pygame.draw.circle(surface, (198, 207, 215), center, radius, 1)
            if self.cell >= 23:
                label = (
                    str(self.human_ids.index(people[0]) + 1)
                    if len(people) == 1
                    else f"{len(people)}x"
                )
                image = self.font(12, True).render(label, True, (146, 158, 169))
                surface.blit(image, image.get_rect(center=center))

    def button(
        self, surface: pygame.Surface, key: str, label: str, rect: tuple, active: bool = False
    ) -> None:
        box = pygame.Rect(rect)
        self.buttons[key] = box
        pygame.draw.rect(surface, BLUE if active else (232, 238, 245), box, border_radius=8)
        image = self.font(15, True).render(
            self.fit_text(label, box.width - 16, 15, True), True, PANEL if active else INK
        )
        surface.blit(image, image.get_rect(center=box.center))

    def fit_text(self, value: str, width: int, size: int, bold: bool = False) -> str:
        font = self.font(size, bold)
        if font.size(value)[0] <= width:
            return value
        while value and font.size(value + "...")[0] > width:
            value = value[:-1]
        return value + "..."

    def _draw_batch_bar(self, surface: pygame.Surface) -> None:
        if self.catalog is None or not self.catalog.is_batch:
            return
        entry = self.catalog.entries[self.replay_index]
        pygame.draw.rect(surface, PANEL, (22, 100, self.width - 44, 80), border_radius=14)
        for dimension, label, value, x, width in (
            ("map", "MAP", self.catalog.maps[entry.map_key], 38, 450),
            ("behavior", "PEDESTRIANS", entry.behavior, 504, 205),
            ("planner", "ROBOT", entry.planner, 725, 170),
        ):
            self.text(surface, label, (x, 110), 11, MUTED, True)
            self.button(
                surface,
                f"select_{dimension}",
                value + "  v",
                (x, 131, width, 33),
                self.dropdown == dimension,
            )
        self.button(surface, "previous_replay", "< Replay", (915, 131, 130, 33))
        self.button(surface, "next_replay", "Replay >", (1055, 131, 165, 33))

    def _draw_dropdown(self, surface: pygame.Surface) -> None:
        self.menu_boxes = []
        if self.dropdown is None or self.catalog is None:
            return
        options = self.catalog.options(self.dropdown, self.replay_index)
        anchor = self.buttons[f"select_{self.dropdown}"]
        visible = min(10, len(options), (self.height - anchor.bottom - 48) // 32)
        self.menu_scroll = max(0, min(self.menu_scroll, len(options) - visible))
        footer = 22 if len(options) > visible else 0
        popup = pygame.Rect(anchor.x, anchor.bottom + 5, anchor.width, visible * 32 + 8 + footer)
        pygame.draw.rect(surface, (211, 220, 229), popup.move(3, 4), border_radius=9)
        pygame.draw.rect(surface, PANEL, popup, border_radius=9)
        pygame.draw.rect(surface, GRID_LINE, popup, 1, border_radius=9)
        current = self.catalog.entries[self.replay_index]
        selected = {
            "map": current.map_key,
            "behavior": current.behavior,
            "planner": current.planner,
        }
        for row, (value, label) in enumerate(
            options[self.menu_scroll : self.menu_scroll + visible]
        ):
            rect = pygame.Rect(popup.x + 4, popup.y + 4 + row * 32, popup.width - 8, 32)
            if value == selected[self.dropdown]:
                pygame.draw.rect(surface, (224, 235, 255), rect, border_radius=5)
            self.text(
                surface,
                self.fit_text(label, rect.width - 18, 14),
                (rect.x + 9, rect.y + 6),
                14,
                BLUE if value == selected[self.dropdown] else INK,
            )
            self.menu_boxes.append((value, rect))
        if footer:
            self.text(
                surface,
                f"Scroll for more  /  {self.menu_scroll + 1}-{self.menu_scroll + visible}"
                f" of {len(options)}",
                (popup.x + 12, popup.bottom - 22),
                11,
                MUTED,
            )

    def handle_click(self, position: tuple[int, int]) -> None:
        if self.show_results:
            for action, rect in self.buttons.items():
                if rect.collidepoint(position):
                    self.action(action)
                    break
            return
        if self.dropdown is not None and self.catalog is not None:
            for value, rect in self.menu_boxes:
                if rect.collidepoint(position):
                    index = self.catalog.select(self.dropdown, value, self.replay_index)
                    self.switch_replay(index)
                    return
            for action, rect in self.buttons.items():
                if action.startswith("select_") and rect.collidepoint(position):
                    self.action(action)
                    return
            self.dropdown = None
            return  # Closing a popup must not activate a control underneath it.
        if hasattr(self, "timeline") and self.timeline.inflate(12, 20).collidepoint(position):
            ratio = (position[0] - self.timeline.x) / self.timeline.width
            self.playhead = float(round(max(0, min(1, ratio)) * self.last_tick))
            self.playing = False
            return
        for action, rect in self.buttons.items():
            if rect.collidepoint(position):
                self.action(action)
                return

    def viewport(self, window_size: tuple[int, int]) -> pygame.Rect:
        """Fit the complete interface without stretching grid cells or cropping controls."""
        window_width, window_height = window_size
        scale = min(max(1, window_width) / self.width, max(1, window_height) / self.height)
        width, height = max(1, round(self.width * scale)), max(1, round(self.height * scale))
        return pygame.Rect(
            (window_width - width) // 2, (window_height - height) // 2, width, height
        )

    def window_to_canvas(
        self,
        position: tuple[int, int],
        window_size: tuple[int, int],
    ) -> tuple[int, int] | None:
        viewport = self.viewport(window_size)
        if not viewport.collidepoint(position):
            return None
        return (
            int((position[0] - viewport.x) * self.width / viewport.width),
            int((position[1] - viewport.y) * self.height / viewport.height),
        )

    def handle_window_click(self, position: tuple[int, int], window_size: tuple[int, int]) -> None:
        mapped = self.window_to_canvas(position, window_size)
        if mapped is not None:
            self.handle_click(mapped)

    def draw_window(self, surface: pygame.Surface) -> None:
        self.window_size = surface.get_size()
        canvas = pygame.Surface((self.width, self.height))
        self.draw(canvas)
        viewport = self.viewport(self.window_size)
        surface.fill(BACKGROUND)
        if viewport.size != canvas.get_size():
            canvas = pygame.transform.smoothscale(canvas, viewport.size)
        surface.blit(canvas, viewport.topleft)

    @staticmethod
    def initial_window_size(desktop_size: tuple[int, int]) -> tuple[int, int]:
        return (max(1, min(1100, desktop_size[0] - 80)), max(1, min(760, desktop_size[1] - 100)))

    def _draw_map(self, surface: pygame.Surface) -> None:
        card = pygame.Rect(
            22,
            100 + self.content_offset,
            self.sidebar_x - 38,
            self.height - 223 - self.content_offset,
        )
        pygame.draw.rect(surface, PANEL, card, border_radius=14)
        for y, row in enumerate(self.rows):
            for x, value in enumerate(row):
                rect = pygame.Rect(
                    self.map_x + x * self.cell, self.map_y + y * self.cell, self.cell, self.cell
                )
                pygame.draw.rect(surface, WALL if value == "#" else (247, 250, 252), rect)
                if self.cell >= 10:
                    pygame.draw.rect(surface, GRID_LINE if value == "." else (77, 94, 110), rect, 1)
        if self.cell >= 16:
            for x in range(len(self.rows[0])):
                self.text(
                    surface, str(x), (self.map_x + x * self.cell + 5, self.map_y - 21), 11, MUTED
                )
            for y in range(len(self.rows)):
                self.text(
                    surface, str(y), (self.map_x - 23, self.map_y + y * self.cell + 5), 11, MUTED
                )
        # Arrived humans belong to the floor layer. Paths, halos, and active
        # agents are drawn over them, including when reusing the same goal cell.
        self._draw_arrived_humans(surface)
        # Start and goal markers remain visible after the robot leaves.
        for p, label in (
            (self.scenario["robot"]["start"], "S"),
            (self.scenario["robot"]["goals"][-1], "G"),
        ):
            center = self.center(p)
            pygame.draw.circle(surface, (207, 222, 254), center, max(2, self.cell // 3), 2)
            self.text(surface, label, (center[0] - 5, center[1] - 10), 15, BLUE, True)

        if self.show_path:
            points = [
                self.center(frame["positions"][self.robot_id])
                for frame in self.frames
                if self.robot_id in frame["positions"]
            ]
            if self.metrics["robot_completed"]:
                points.append(self.center(self.scenario["robot"]["goals"][-1]))
            if len(points) > 1:
                pygame.draw.lines(surface, (178, 200, 248), False, points, 3)

        if self.show_baseline:
            for p in self.interpolated(self.baseline, self.playhead).values():
                pygame.draw.circle(
                    surface, (170, 181, 192), self.center(p), max(2, int(self.cell * 0.31)), 2
                )

        positions = self.interpolated(self.frames, self.playhead)
        if self.show_halos:
            overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
            clip = pygame.Rect(
                self.map_x, self.map_y, len(self.rows[0]) * self.cell, len(self.rows) * self.cell
            )
            overlay.set_clip(clip)
            for ident, p in positions.items():
                if ident == self.robot_id:
                    continue
                radius = round(self.specs[ident]["personal_radius"] * self.cell)
                color = self.colors[ident]
                pygame.draw.circle(overlay, (*color, 21), self.center(p), radius)
                pygame.draw.circle(overlay, (*color, 65), self.center(p), radius, 1)
            surface.blit(overlay, (0, 0))

        for ident, p in positions.items():
            center = self.center(p)
            radius = max(3, round(self.cell * 0.29))
            if ident == self.robot_id:
                box = pygame.Rect(0, 0, radius * 2 + 4, radius * 2 + 4)
                box.center = center
                pygame.draw.rect(surface, BLUE, box, border_radius=max(2, radius // 2))
                if self.cell >= 18:
                    for delta in (-4, 4):
                        pygame.draw.circle(surface, PANEL, (center[0] + delta, center[1] - 2), 2)
                    pygame.draw.line(
                        surface,
                        PANEL,
                        (center[0] - 4, center[1] + 5),
                        (center[0] + 4, center[1] + 5),
                        2,
                    )
            else:
                color = self.colors[ident]
                pygame.draw.circle(surface, color, center, radius)
                pygame.draw.circle(surface, PANEL, center, radius, 2)
                if self.cell >= 23:
                    label = str(self.human_ids.index(ident) + 1)
                    image = self.font(13, True).render(label, True, PANEL)
                    surface.blit(image, image.get_rect(center=center))

    def _draw_sidebar(self, surface: pygame.Surface) -> None:
        x = self.sidebar_x
        offset = self.content_offset
        pygame.draw.rect(
            surface, PANEL, (x, 100 + offset, 294, self.height - 223 - offset), border_radius=14
        )
        self.text(surface, "RUN DETAILS", (x + 20, 118 + offset), 12, MUTED, True)
        self.text(surface, self.metrics["planner"], (x + 20, 142 + offset), 29, BLUE, True)
        self.text(surface, f"Pedestrians: {self.metrics['behavior']}", (x + 20, 182 + offset), 15)
        self.text(
            surface,
            f"{len(self.human_ids)} people  /  A* walking routes",
            (x + 20, 206 + offset),
            14,
            MUTED,
        )
        pygame.draw.line(surface, GRID_LINE, (x + 20, 240 + offset), (x + 274, 240 + offset))
        self.text(surface, "FINAL EVALUATION", (x + 20, 257 + offset), 12, MUTED, True)
        values = [
            (
                "Robot arrival",
                str(self.metrics["robot_arrival_time"])
                if self.metrics["robot_completed"]
                else "unfinished",
            ),
            ("Total human delay", f"{self.metrics['total_pedestrian_delay']} ticks"),
            ("Largest human delay", f"{self.metrics['max_pedestrian_delay']} ticks"),
            ("People arrived", f"{self.metrics['pedestrians_completed']}/{len(self.human_ids)}"),
            ("Space exposure", f"{self.metrics['personal_space_exposure']:.2f}"),
        ]
        for index, (label, value) in enumerate(values):
            y = 287 + offset + index * 33
            self.text(surface, label, (x + 20, y), 14, MUTED)
            image = self.font(15, True).render(value, True, INK)
            surface.blit(image, (x + 274 - image.get_width(), y))
        self.text(
            surface, "Matched against a run without the robot.", (x + 20, 465 + offset), 12, MUTED
        )
        if self.metrics["pedestrians_completed"] < len(self.human_ids):
            self.text(
                surface,
                "Unfinished trips: delay is horizon-limited.",
                (x + 20, 486 + offset),
                12,
                (180, 101, 42),
            )
        elif self.metrics["baseline_completed"] < len(self.human_ids):
            self.text(
                surface,
                "Baseline has unfinished pedestrian trips.",
                (x + 20, 486 + offset),
                12,
                (180, 101, 42),
            )
        self.button(
            surface, "halos", "H  Personal space", (x + 18, 522 + offset, 258, 30), self.show_halos
        )
        self.button(
            surface,
            "baseline",
            "B  Baseline ghosts",
            (x + 18, 560 + offset, 258, 30),
            self.show_baseline,
        )
        self.button(
            surface,
            "path",
            "P  Executed robot route",
            (x + 18, 598 + offset, 258, 30),
            self.show_path,
        )

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(BACKGROUND)
        self.buttons.clear()
        if self.show_results and self.explorer is not None:
            self.explorer.draw(surface, self)
            self.button(surface, "results", "Back to replay", (self.width - 329, 32, 153, 35))
            self.button(surface, "open_batch", "Open batch...", (self.width - 166, 32, 138, 35))
            if self.notice:
                self.text(
                    surface,
                    self.fit_text(self.notice, self.width - 60, 12),
                    (29, 93),
                    12,
                    (180, 72, 42),
                )
            return
        self.text(surface, "HUMAN-AWARE NAVIGATION", (28, 17), 12, GREEN, True)
        self.text(
            surface,
            self.fit_text(self.scenario["name"], self.width - 400, 27, True),
            (28, 39),
            27,
            INK,
            True,
        )
        self.button(surface, "open_batch", "Open batch...", (self.width - 166, 32, 138, 35))
        self.button(surface, "results", "3D results", (self.width - 329, 32, 153, 35))
        self.text(
            surface,
            "Recorded simulation  /  smooth playback  /  fixed discrete timesteps"
            + (
                f"  /  Replay {self.replay_index + 1}/{len(self.catalog.entries)}"
                if self.catalog
                else ""
            ),
            (29, 76),
            13,
            MUTED,
        )
        self._draw_batch_bar(surface)
        self._draw_map(surface)
        self._draw_sidebar(surface)
        y = self.height - 106
        self.button(surface, "play", "Pause" if self.playing else "Play", (28, y, 86, 36), True)
        self.button(surface, "previous", "<", (122, y, 38, 36))
        self.button(surface, "next", ">", (166, y, 38, 36))
        self.button(surface, "reset", "Reset", (212, y, 66, 36))
        self.button(surface, "slower", "-", (294, y, 35, 36))
        self.text(surface, f"{self.speed:g}x", (341, y + 6), 17, INK, True)
        self.button(surface, "faster", "+", (399, y, 35, 36))
        self.text(
            surface,
            f"Tick {int(self.playhead):03d} / {self.last_tick:03d}",
            (self.width - 192, y + 7),
            17,
            INK,
            True,
        )
        self.timeline = pygame.Rect(459, y + 13, self.width - 675, 10)
        pygame.draw.rect(surface, (210, 221, 232), self.timeline, border_radius=5)
        fraction = self.playhead / max(1, self.last_tick)
        pygame.draw.rect(
            surface,
            BLUE,
            (*self.timeline.topleft, int(self.timeline.width * fraction), 10),
            border_radius=5,
        )
        pygame.draw.circle(
            surface, BLUE, (round(self.timeline.x + self.timeline.width * fraction), y + 18), 7
        )
        self.text(
            surface,
            "Space: play/pause     Left / Right: one timestep     Up / Down: speed"
            "     PgUp / PgDn: replay     Esc: close",
            (29, self.height - 47),
            13,
            MUTED,
        )
        self.text(
            surface,
            "Playback speed changes animation only. Metrics use simulation ticks.",
            (29, self.height - 27),
            12,
            MUTED,
        )
        if self.notice:
            self.text(
                surface,
                self.fit_text(self.notice, self.width - 60, 12),
                (29, 185 if self.content_offset else 84),
                12,
                (180, 72, 42),
            )
        self._draw_dropdown(surface)

    def action(self, action: str) -> None:
        if action == "results":
            if self.explorer is None:
                from .results_gui import ResultsExplorer

                try:
                    self.explorer = (
                        ResultsExplorer.from_catalog(self.catalog)
                        if self.catalog
                        else ResultsExplorer.from_replay(self.replay)
                    )
                    if self.metrics["behavior"] in self.explorer.data.behaviors:
                        self.explorer.behavior = self.metrics["behavior"]
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    self.notice = f"Could not open results: {exc}"
                    return
            self.show_results = not self.show_results
            self.dropdown = None
            self.explorer.drag = None
        elif action.startswith("graph_") and self.explorer is not None:
            try:
                self.explorer.action(action.removeprefix("graph_"))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.notice = f"Could not update results: {exc}"
        elif action == "open_batch":
            self._choose_batch_folder()
        elif action in {"next_replay", "previous_replay"}:
            self.switch_replay(self.replay_index + (1 if action == "next_replay" else -1))
        elif action.startswith("select_") and self.catalog is not None:
            dimension = action.removeprefix("select_")
            self.dropdown = None if self.dropdown == dimension else dimension
            self.menu_scroll = 0
        elif action == "play":
            if self.playhead >= self.last_tick:
                self.playhead = 0.0
            self.playing = not self.playing
        elif action in {"next", "previous"}:
            self.playing = False
            self.playhead = float(
                max(0, min(self.last_tick, int(self.playhead) + (1 if action == "next" else -1)))
            )
        elif action == "reset":
            self.playhead = 0.0
        elif action == "faster":
            self.speed = min(16.0, self.speed * 2)
        elif action == "slower":
            self.speed = max(0.125, self.speed / 2)
        elif action == "halos":
            self.show_halos = not self.show_halos
        elif action == "baseline":
            self.show_baseline = not self.show_baseline
        elif action == "path":
            self.show_path = not self.show_path

    def run(self) -> None:
        pygame.display.init()
        pygame.font.init()
        desktop = pygame.display.get_desktop_sizes()[0]
        self.window_size = self.window_size or self.initial_window_size(desktop)
        screen = pygame.display.set_mode(self.window_size, pygame.RESIZABLE)
        self._update_caption()
        clock = pygame.time.Clock()
        keys = {
            pygame.K_SPACE: "play",
            pygame.K_RIGHT: "next",
            pygame.K_LEFT: "previous",
            pygame.K_HOME: "reset",
            pygame.K_UP: "faster",
            pygame.K_DOWN: "slower",
            pygame.K_h: "halos",
            pygame.K_b: "baseline",
            pygame.K_p: "path",
            pygame.K_PAGEUP: "previous_replay",
            pygame.K_PAGEDOWN: "next_replay",
        }
        running = True
        while running:
            # Opening a native folder dialog or loading a large trace should
            # not make playback jump to the end when the event loop resumes.
            seconds = min(clock.tick(60) / 1000.0, 0.1)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.show_results:
                            self.action("results")
                        elif self.dropdown is not None:
                            self.dropdown = None
                        else:
                            running = False
                    elif self.show_results:
                        graph_keys = {
                            pygame.K_1: "face_front",
                            pygame.K_2: "face_side",
                            pygame.K_3: "face_top",
                            pygame.K_s: "slices",
                            pygame.K_r: "reset",
                        }
                        if event.key in graph_keys:
                            self.action("graph_" + graph_keys[event.key])
                    elif event.key in keys:
                        self.action(keys[event.key])
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    position = self.window_to_canvas(
                        event.pos, pygame.display.get_surface().get_size()
                    )
                    handled = False
                    if self.show_results and position is not None:
                        handled = self.explorer.pointer_down(
                            event.button, position, bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)
                        )
                    if event.button == 1 and position is not None and not handled:
                        self.handle_click(position)
                elif event.type == pygame.MOUSEMOTION and self.show_results:
                    self.explorer.pointer_move(
                        self.window_to_canvas(event.pos, pygame.display.get_surface().get_size())
                    )
                elif event.type == pygame.MOUSEBUTTONUP and self.explorer is not None:
                    self.explorer.drag = None
                elif event.type == pygame.MOUSEWHEEL:
                    if self.show_results:
                        position = self.window_to_canvas(
                            pygame.mouse.get_pos(), pygame.display.get_surface().get_size()
                        )
                        if position is not None and self.explorer.plot_rect.collidepoint(position):
                            self.explorer.wheel(event.y)
                    elif self.dropdown is not None:
                        self.menu_scroll -= event.y
                elif event.type == pygame.WINDOWFOCUSLOST and self.explorer is not None:
                    self.explorer.drag = None
                elif event.type == pygame.VIDEORESIZE:
                    if self.explorer is not None:
                        self.explorer.drag = None
                    self.window_size = (max(1, event.w), max(1, event.h))
                    pygame.display.set_mode(self.window_size, pygame.RESIZABLE)
            if self.show_results:
                self.explorer.camera.update(seconds)
            elif self.playing and self.dropdown is None:
                step_duration = self.scenario["settings"]["seconds_per_step"]
                self.playhead = min(
                    self.last_tick, self.playhead + seconds * self.speed / step_duration
                )
                if self.playhead >= self.last_tick:
                    self.playing = False
            screen = pygame.display.get_surface()
            self.draw_window(screen)
            pygame.display.flip()
        pygame.quit()

    def screenshot(self, path: str | Path, tick: float = 0.0) -> None:
        pygame.font.init()
        self.playhead = max(0.0, min(float(tick), self.last_tick))
        self.playing = False
        surface = pygame.Surface((self.width, self.height))
        self.draw(surface)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(surface, str(path))
