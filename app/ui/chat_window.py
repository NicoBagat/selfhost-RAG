"""
app/ui/chat_window.py

PyQt6 chat interface for the Obsidian RAG assistant.

Architecture
------------
qasync merges the asyncio event loop with the Qt event loop so that
Orchestrator.handle_query() — which internally uses asyncio.to_thread for
streaming — runs as a plain asyncio.Task on the main thread.  No QThread
or extra synchronisation is required.

Token delivery
--------------
chat_stream() runs in a thread pool (asyncio.to_thread in the orchestrator).
Each token is posted back to the event loop via loop.call_soon_threadsafe,
which schedules ChatWindow._on_token() on the Qt/asyncio main thread.
_on_token() is therefore always called on the UI thread and can update
QTextEdit directly without further locking.

Components
----------
ChatDisplay  (QTextEdit)   — read-only history; streams tokens into the
                             current assistant block via a saved QTextCursor.
InputBar     (QWidget)     — QPlainTextEdit + Send/Cancel QPushButton.
ChatWindow   (QMainWindow) — root window; owns the Orchestrator reference
                             and drives the async query lifecycle.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QKeySequence, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from app.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Palette:
    window_bg: str
    display_bg: str
    input_bg: str
    input_border: str
    body: str
    separator: str
    user_label: str
    asst_label: str
    err_label: str
    btn_send: str
    btn_send_hover: str
    btn_send_pressed: str
    btn_send_disabled: str
    btn_cancel: str
    btn_cancel_hover: str


_LIGHT = _Palette(
    window_bg="#f3f4f6",
    display_bg="#f9fafb",
    input_bg="#ffffff",
    input_border="#d1d5db",
    body="#1a1a1a",
    separator="#9ca3af",
    user_label="#2563eb",
    asst_label="#16a34a",
    err_label="#dc2626",
    btn_send="#2563eb",
    btn_send_hover="#1d4ed8",
    btn_send_pressed="#1e40af",
    btn_send_disabled="#93c5fd",
    btn_cancel="#dc2626",
    btn_cancel_hover="#b91c1c",
)

_DARK = _Palette(
    window_bg="#181825",
    display_bg="#1e1e2e",
    input_bg="#313244",
    input_border="#45475a",
    body="#cdd6f4",
    separator="#45475a",
    user_label="#89b4fa",
    asst_label="#a6e3a1",
    err_label="#f38ba8",
    btn_send="#89b4fa",
    btn_send_hover="#74c7ec",
    btn_send_pressed="#89dceb",
    btn_send_disabled="#45475a",
    btn_cancel="#f38ba8",
    btn_cancel_hover="#eba0ac",
)


# ---------------------------------------------------------------------------
# ChatDisplay
# ---------------------------------------------------------------------------


class ChatDisplay(QTextEdit):
    """
    Read-only message history.

    User messages and assistant messages use distinct label colours.
    Assistant messages are built incrementally: begin_assistant_message()
    creates the block and stores a cursor; append_token() inserts into it;
    end_assistant_message() closes the block.
    """

    def __init__(self, palette: _Palette, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette
        self.setReadOnly(True)
        self.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)

        font = QFont("Inter", 11)
        font.setStyleHint(QFont.StyleHint.SansSerif)
        self.setFont(font)
        self.setStyleSheet(
            f"QTextEdit {{"
            f"  background-color: {palette.display_bg};"
            f"  color: {palette.body};"
            f"  border: none;"
            f"  padding: 8px;"
            f"}}"
        )

        self._stream_cursor: QTextCursor | None = None

    # ------------------------------------------------------------------
    # User message
    # ------------------------------------------------------------------

    def add_user_message(self, text: str) -> None:
        cursor = self._end_cursor()
        self._insert_separator(cursor)
        self._insert_label(cursor, "You", QColor(self._palette.user_label))
        self._insert_body(cursor, text)
        self._insert_newline(cursor)
        self.ensureCursorVisible()

    # ------------------------------------------------------------------
    # Assistant message (streaming)
    # ------------------------------------------------------------------

    def begin_assistant_message(self) -> None:
        """Open an assistant block and park the stream cursor at its end."""
        cursor = self._end_cursor()
        self._insert_separator(cursor)
        self._insert_label(cursor, "Assistant", QColor(self._palette.asst_label))

        fmt = QTextCharFormat()
        fmt.setForeground(QColor(self._palette.body))
        fmt.setFontWeight(QFont.Weight.Normal)
        cursor.setCharFormat(fmt)
        self._stream_cursor = cursor
        self.ensureCursorVisible()

    def append_token(self, token: str) -> None:
        """Append a streamed token to the current assistant block."""
        if self._stream_cursor is None:
            return
        self._stream_cursor.insertText(token)
        self.ensureCursorVisible()

    def end_assistant_message(self) -> None:
        """Close the current assistant block."""
        if self._stream_cursor is not None:
            self._stream_cursor.insertText("\n")
            self._stream_cursor = None

    # ------------------------------------------------------------------
    # Error / system messages
    # ------------------------------------------------------------------

    def add_error_message(self, text: str) -> None:
        cursor = self._end_cursor()
        self._insert_separator(cursor)
        self._insert_label(cursor, "Error", QColor(self._palette.err_label))
        self._insert_body(cursor, text)
        self._insert_newline(cursor)
        self.ensureCursorVisible()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _end_cursor(self) -> QTextCursor:
        cursor = QTextCursor(self.document())
        cursor.movePosition(QTextCursor.MoveOperation.End)
        return cursor

    def _insert_label(
        self, cursor: QTextCursor, label: str, colour: QColor
    ) -> None:
        fmt = QTextCharFormat()
        fmt.setFontWeight(QFont.Weight.Bold)
        fmt.setForeground(colour)
        fmt.setFontPointSize(11)
        cursor.insertText(f"{label}\n", fmt)

    def _insert_body(self, cursor: QTextCursor, text: str) -> None:
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(self._palette.body))
        fmt.setFontWeight(QFont.Weight.Normal)
        cursor.insertText(text, fmt)

    def _insert_separator(self, cursor: QTextCursor) -> None:
        if cursor.position() == 0:
            return
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(self._palette.separator))
        cursor.insertText("\n\u2500\u2500\u2500\n", fmt)  # ─── rule

    def _insert_newline(self, cursor: QTextCursor) -> None:
        fmt = QTextCharFormat()
        cursor.insertText("\n", fmt)


# ---------------------------------------------------------------------------
# InputBar
# ---------------------------------------------------------------------------


class InputBar(QWidget):
    """
    Single-line (Enter to send) or multi-line (Shift+Enter for newline)
    text entry with a Send / Cancel button.
    """

    message_submitted = pyqtSignal(str)

    def __init__(self, palette: _Palette, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._palette = palette

        self._input = QPlainTextEdit()
        self._input.setFixedHeight(60)
        self._input.setPlaceholderText("Ask something about your vault… (Enter to send)")
        self._input.setStyleSheet(
            f"QPlainTextEdit {{"
            f"  border: 1px solid {palette.input_border};"
            f"  border-radius: 6px;"
            f"  padding: 6px;"
            f"  font-size: 11pt;"
            f"  background-color: {palette.input_bg};"
            f"  color: {palette.body};"
            f"}}"
        )

        self._btn = QPushButton("Send")
        self._btn.setFixedSize(80, 60)
        self._btn.setStyleSheet(self._send_stylesheet())
        self._btn.clicked.connect(self._on_send)

        # Enter submits; Shift+Enter inserts a newline
        self._input.installEventFilter(self)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._input)
        layout.addWidget(self._btn)

    def eventFilter(self, obj, event) -> bool:  # type: ignore[override]
        from PyQt6.QtCore import QEvent
        from PyQt6.QtGui import QKeyEvent
        if obj is self._input and event.type() == QEvent.Type.KeyPress:
            key_event: QKeyEvent = event  # type: ignore[assignment]
            if (
                key_event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
                and not (key_event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            ):
                self._on_send()
                return True
        return super().eventFilter(obj, event)

    def _on_send(self) -> None:
        text = self._input.toPlainText().strip()
        if text:
            self._input.clear()
            self.message_submitted.emit(text)

    def set_busy(self, busy: bool) -> None:
        """Toggle between Send (idle) and Cancel (busy) states."""
        if busy:
            self._btn.setText("Cancel")
            self._btn.setStyleSheet(self._cancel_stylesheet())
            self._input.setEnabled(False)
        else:
            self._btn.setText("Send")
            self._btn.setStyleSheet(self._send_stylesheet())
            self._input.setEnabled(True)
            self._input.setFocus()

    def _send_stylesheet(self) -> str:
        p = self._palette
        return (
            f"QPushButton {{ background-color: {p.btn_send}; color: white;"
            f"  border-radius: 6px; font-weight: bold; font-size: 11pt; }}"
            f"QPushButton:hover   {{ background-color: {p.btn_send_hover}; }}"
            f"QPushButton:pressed {{ background-color: {p.btn_send_pressed}; }}"
            f"QPushButton:disabled {{ background-color: {p.btn_send_disabled}; }}"
        )

    def _cancel_stylesheet(self) -> str:
        p = self._palette
        return (
            f"QPushButton {{ background-color: {p.btn_cancel}; color: white;"
            f"  border-radius: 6px; font-weight: bold; font-size: 11pt; }}"
            f"QPushButton:hover {{ background-color: {p.btn_cancel_hover}; }}"
        )


# ---------------------------------------------------------------------------
# ChatWindow
# ---------------------------------------------------------------------------


class ChatWindow(QMainWindow):
    """
    Root application window.

    Constructs the layout, owns the Orchestrator reference, and drives the
    async query lifecycle via asyncio.ensure_future().

    Parameters
    ----------
    orchestrator : Orchestrator
        Fully constructed orchestrator.  MCPClient.start() (if applicable)
        should be called before or shortly after the window is shown —
        _startup() schedules it automatically if mcp is attached.
    title : str
        Window title.
    """

    def __init__(
        self,
        orchestrator: Orchestrator,
        title: str = "Obsidian RAG",
        theme: str = "light",
    ) -> None:
        super().__init__()
        self._orchestrator = orchestrator
        self._current_task: asyncio.Task | None = None  # type: ignore[type-arg]
        self._palette = _DARK if theme == "dark" else _LIGHT

        self.setWindowTitle(title)
        self.resize(860, 640)
        self._build_ui()
        self._connect_signals()

        # Schedule async startup (MCP connection) after the event loop starts
        asyncio.ensure_future(self._startup())

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 8)
        layout.setSpacing(8)

        self._display = ChatDisplay(self._palette)
        self._display.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self._input_bar = InputBar(self._palette)
        self._input_bar.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        layout.addWidget(self._display)
        layout.addWidget(self._input_bar)

        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready")

        toolbar = QToolBar("Actions")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # --- mode toggle (RAG / Auto / MCP) ---
        toolbar.addWidget(QLabel("  Mode: "))
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._btn_rag  = self._make_toggle_btn("RAG")
        self._btn_auto = self._make_toggle_btn("Auto")
        self._btn_mcp  = self._make_toggle_btn("MCP")
        for btn in (self._btn_rag, self._btn_auto, self._btn_mcp):
            self._mode_group.addButton(btn)
            toolbar.addWidget(btn)
        self._btn_auto.setChecked(True)

        toolbar.addSeparator()

        # --- re-index button ---
        self._reindex_btn = QPushButton("Re-index vault")
        self._reindex_btn.setStyleSheet(
            f"QPushButton {{ background-color: {self._palette.btn_send}; color: white;"
            f"  border-radius: 4px; padding: 4px 10px; font-size: 10pt; }}"
            f"QPushButton:hover {{ background-color: {self._palette.btn_send_hover}; }}"
            f"QPushButton:disabled {{ background-color: {self._palette.btn_send_disabled}; }}"
        )
        self._reindex_btn.clicked.connect(self._on_reindex_clicked)
        toolbar.addWidget(self._reindex_btn)

        self.setStyleSheet(
            f"QMainWindow {{ background-color: {self._palette.window_bg}; }}"
        )

    def _make_toggle_btn(self, label: str) -> QPushButton:
        p = self._palette
        btn = QPushButton(label)
        btn.setCheckable(True)
        btn.setStyleSheet(
            f"QPushButton {{ background-color: {p.input_bg}; color: {p.body};"
            f"  border: 1px solid {p.input_border}; border-radius: 4px;"
            f"  padding: 4px 10px; font-size: 10pt; }}"
            f"QPushButton:checked {{ background-color: {p.btn_send}; color: white;"
            f"  border-color: {p.btn_send}; }}"
            f"QPushButton:hover {{ background-color: {p.btn_send_hover}; color: white; }}"
        )
        return btn

    def _get_intent_override(self):
        from app.orchestrator import QueryIntent
        if self._btn_rag.isChecked():
            return QueryIntent.RAG
        if self._btn_mcp.isChecked():
            return QueryIntent.MCP
        return None  # Auto

    def _connect_signals(self) -> None:
        self._input_bar.message_submitted.connect(self._on_message_submitted)
        # Cancel on Escape
        from PyQt6.QtGui import QShortcut
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self).activated.connect(
            self._cancel_current_task
        )

    # ------------------------------------------------------------------
    # Async lifecycle
    # ------------------------------------------------------------------

    async def _startup(self) -> None:
        """Start the MCPClient if one is attached to the orchestrator."""
        mcp = getattr(self._orchestrator, "_mcp", None)
        if mcp is None:
            return
        self._status_bar.showMessage("Connecting to mcp-obsidian…")
        try:
            await mcp.start()
            tool_count = len(mcp.tools)
            self._status_bar.showMessage(
                f"Ready — {tool_count} MCP tool(s) available"
            )
            logger.info("MCPClient started with %d tool(s)", tool_count)
        except Exception as exc:
            self._status_bar.showMessage(f"MCP offline: {exc}")
            logger.warning("MCPClient failed to start: %s", exc)

    async def _process_query(self, query: str) -> None:
        """
        Run a full query/response cycle.

        Called via asyncio.ensure_future so it does not block the Qt event
        loop.  Streams tokens into ChatDisplay as they arrive.
        """
        self._display.begin_assistant_message()
        self._status_bar.showMessage("Thinking…")

        override = self._get_intent_override()
        intent_used = None
        try:
            intent_used = await self._orchestrator.handle_query(
                query, self._on_token, intent_override=override
            )
        except asyncio.CancelledError:
            self._display.append_token("\n[cancelled]")
            logger.info("Query cancelled by user")
        except Exception as exc:
            self._display.add_error_message(str(exc))
            logger.error("Query failed: %s", exc)
        finally:
            self._display.end_assistant_message()
            self._input_bar.set_busy(False)
            if intent_used is not None:
                mode_label = "auto" if override is None else "manual"
                self._status_bar.showMessage(
                    f"Ready — last query: {intent_used.name} ({mode_label})"
                )
            else:
                self._status_bar.showMessage("Ready")
            self._current_task = None

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_message_submitted(self, text: str) -> None:
        if self._current_task and not self._current_task.done():
            # Second click while busy → cancel
            self._cancel_current_task()
            return

        self._display.add_user_message(text)
        self._input_bar.set_busy(True)
        self._current_task = asyncio.ensure_future(self._process_query(text))

    def _on_token(self, token: str) -> None:
        """Receive a streamed token from the orchestrator thread pool."""
        self._display.append_token(token)

    def _cancel_current_task(self) -> None:
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            logger.debug("Current query task cancelled")

    def _on_reindex_clicked(self) -> None:
        asyncio.ensure_future(self._run_reindex())

    async def _run_reindex(self) -> None:
        self._reindex_btn.setEnabled(False)
        self._status_bar.showMessage("Re-indexing vault…")
        try:
            stats = await self._orchestrator.reindex()
            self._status_bar.showMessage(
                f"Re-index complete — {stats.indexed} indexed, "
                f"{stats.skipped} skipped, {stats.errors} errors"
            )
        except Exception as exc:
            self._status_bar.showMessage(f"Re-index failed: {exc}")
            logger.error("Re-index failed: %s", exc)
        finally:
            self._reindex_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Close event
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # type: ignore[override]
        """Cancel any running task and stop the MCP client on window close."""
        self._cancel_current_task()
        mcp = getattr(self._orchestrator, "_mcp", None)
        if mcp and mcp.is_running:
            # Schedule async stop; the event loop will process it before exit
            asyncio.ensure_future(mcp.stop())
        super().closeEvent(event)
