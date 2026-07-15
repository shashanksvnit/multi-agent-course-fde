"""
agent.py  -  the "brain" (Layer B). LLM + tool loop over a Provider.

Tools mirror a hotel reservations desk:
    check_availability -> find matching rooms
    create_booking     -> reserve a room
    transfer_to_human  -> front desk / human queue
    end_call           -> caller done (real system: SIP BYE)

Uses OpenAI-style function calling, which both Groq and OpenAI support, so this
file is provider-agnostic  -  it only talks to Provider.chat().
"""

from __future__ import annotations

import json

from providers import Provider

SYSTEM_PROMPT = """You are a friendly phone reservations agent for Aurora Hotel.
Your only job is hotel room booking support: new reservations, availability,
room options, rates returned by tools, changing/canceling reservations, and
transferring to the front desk.

Guardrails:
- Do not answer questions outside hotel booking support, including weather,
  news, trivia, coding, medical, legal, finance, or general assistant tasks.
- For off-topic requests, politely say you can only help with hotel reservations
  and ask whether they want to book, change, or cancel a stay.
- Never invent availability, rates, confirmation numbers, policies, or guest
  details. Use tools for availability and booking.
- Keep replies short and spoken-friendly: one or two sentences, no bullet lists,
  no markdown, no emoji.

Booking flow:
1. First collect only check-in date, check-out date, guest count, and optional
   room type preference.
2. Once dates and guests are known, call check_availability immediately, even
   if no room type preference was given.
3. Offer the available room options and ask which one they want.
4. Only after the caller chooses or confirms a room, collect guest name and
   phone or email.
5. Before booking, summarize the selected room and ask for confirmation.
6. After the caller confirms and required details are present, call create_booking.
7. If the caller asks for a person or the request is outside what you can do,
   call transfer_to_human. When the conversation is clearly over, call end_call."""

# OpenAI-style tool schema (works on Groq too).
TOOLS = [
    {
        "type": "function",
            "function": {
            "name": "check_availability",
            "description": "Check hotel room availability for dates, guests, and optional room type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "check_in": {
                        "type": "string",
                        "description": "Check-in date as stated by the caller.",
                    },
                    "check_out": {
                        "type": "string",
                        "description": "Check-out date as stated by the caller.",
                    },
                    "guests": {
                        "type": "integer",
                        "description": "Number of guests.",
                    },
                    "room_type": {
                        "type": "string",
                        "description": "Optional preference: standard, king, suite, family, or accessible.",
                    },
                },
                "required": ["check_in", "check_out", "guests"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_booking",
            "description": "Create a hotel booking after the caller confirms the room option.",
            "parameters": {
                "type": "object",
                "properties": {
                    "check_in": {"type": "string"},
                    "check_out": {"type": "string"},
                    "guests": {"type": "integer"},
                    "room_type": {"type": "string"},
                    "guest_name": {"type": "string"},
                    "contact": {
                        "type": "string",
                        "description": "Phone number or email for the booking.",
                    },
                },
                "required": [
                    "check_in",
                    "check_out",
                    "guests",
                    "room_type",
                    "guest_name",
                    "contact",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_to_human",
            "description": "Hand the call to a human agent queue. Use when the caller "
                           "asks for a person or the request is out of scope.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "end_call",
            "description": "End the call politely when the conversation is finished.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# --- Mock tool implementations (swap for real backends in production) ---

_ROOMS = {
    "standard": {"name": "Standard Queen", "rate": "$189/night", "capacity": 2},
    "king": {"name": "Deluxe King", "rate": "$229/night", "capacity": 2},
    "suite": {"name": "Harbor Suite", "rate": "$329/night", "capacity": 4},
    "family": {"name": "Family Double Queen", "rate": "$269/night", "capacity": 5},
    "accessible": {"name": "Accessible Queen", "rate": "$199/night", "capacity": 2},
}


def _normalize_room_type(value: str | None) -> str | None:
    room_type = (value or "").strip().lower()
    if not room_type:
        return None
    for key in _ROOMS:
        if key in room_type:
            return key
    if "double" in room_type:
        return "family"
    if "queen" in room_type:
        return "standard"
    return None


def run_tool(name: str, args: dict) -> dict:
    """Execute a tool call. The optional 'action' key is a control signal for
    the voice loop ('transfer' -> SIP REFER, 'hangup' -> SIP BYE)."""
    if name == "check_availability":
        guests = int(args.get("guests") or 1)
        preferred = _normalize_room_type(args.get("room_type"))
        rooms = []
        for key, room in _ROOMS.items():
            if preferred and key != preferred:
                continue
            if guests <= room["capacity"]:
                rooms.append(f"{room['name']} at {room['rate']}")
        if not rooms:
            return {
                "result": "No matching rooms are available for that guest count. "
                          "Offer to transfer to the front desk.",
            }
        return {
            "result": "Available rooms for "
                      f"{args.get('check_in')} to {args.get('check_out')}: "
                      f"{'; '.join(rooms)}.",
        }
    if name == "create_booking":
        room_key = _normalize_room_type(args.get("room_type")) or "standard"
        room = _ROOMS[room_key]
        return {
            "result": "Booking confirmed. Confirmation AH-4827 for "
                      f"{args.get('guest_name')} in a {room['name']} from "
                      f"{args.get('check_in')} to {args.get('check_out')} for "
                      f"{args.get('guests')} guest(s). Confirmation sent to "
                      f"{args.get('contact')}.",
        }
    if name == "transfer_to_human":
        return {"result": "Transferring you to the front desk.", "action": "transfer"}
    if name == "end_call":
        return {"result": "Ending the call.", "action": "hangup"}
    return {"result": f"Unknown tool: {name}"}


class Agent:
    """LLM + tool loop for one call. Holds conversation history."""

    def __init__(self, provider: Provider):
        self.provider = provider
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def respond(self, user_text: str) -> tuple[str, str | None]:
        """Take the caller's transcript, return (spoken_reply, action|None).

        Loops until the model produces a plain text reply, executing any tool
        calls in between. `action` is the last control signal seen (transfer/
        hangup), which the voice loop uses to end the call.
        """
        self.messages.append({"role": "user", "content": user_text})
        action: str | None = None

        while True:
            resp = self.provider.chat(self.messages, tools=TOOLS)
            msg = resp.choices[0].message

            if not msg.tool_calls:
                reply = msg.content or ""
                self.messages.append({"role": "assistant", "content": reply})
                return reply, action

            # Record the assistant's tool-call turn, then answer each call.
            self.messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name,
                                     "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = run_tool(tc.function.name, args)
                if result.get("action"):
                    action = result["action"]
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result["result"],
                })
            # loop again so the model can speak given the tool results
