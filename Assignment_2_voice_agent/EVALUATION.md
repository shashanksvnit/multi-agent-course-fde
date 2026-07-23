# Assignment 2 — Evaluation Questions

**Demo video:** https://www.loom.com/share/70f96d1e25f9407eb695347e7810c1f3

Answers below are grounded in what was actually observed running this project end to
end (offline mocks, live typed mode against real GPT-4o-mini, and the LiveKit room
demo with real mic input) — not just the textbook explanation.

---

### Why should the hotel agent refuse a weather question?

The system prompt in `agent.py` scopes the agent to one job: hotel booking support.
Answering a weather question would mean the model is reasoning outside its trusted
context, with no tool to ground the answer — it would just be generating plausible
text. Verified directly: `smoke_test.py` sends *"Can you tell me the weather?"* and the
agent replies *"I can only help with hotel reservations..."* rather than answering. A
narrow, refusing agent is also easier to audit and trust in production than a general
chatbot that happens to also do bookings.

### Why should availability come from a tool instead of the model?

`check_availability` in `agent.py` reads from a fixed `_ROOMS` dict and returns a
deterministic string. If the model were allowed to state availability from its own
reasoning, it could hallucinate a room, a rate, or a date that doesn't exist — with no
way to distinguish a real answer from a plausible-sounding fabrication. Routing through
a tool means the "fact" always traces back to a specific function call with specific
arguments, which is both auditable and swappable (in production, `run_tool` would call
a real PMS/booking-engine API instead of a Python dict, without touching the prompt or
tool schema).

### What happens if STT is wrong?

The transcript is passed to `agent.respond()` verbatim — there's no confidence score or
correction loop. Two outcomes are possible in practice: if the mis-transcription
produces something incoherent, the agent will likely ask a clarifying question (the
prompt's booking flow explicitly asks the caller to confirm the room/dates before
`create_booking` fires, which is the one built-in safety net). But if STT confidently
mishears something plausible-sounding (e.g., "August 12" heard as "August 20"), the
agent has no way to know it's wrong and will book against the wrong date — the
confirmation step only re-states what STT produced, not what the caller actually said.

### Where would barge-in be implemented?

Right now the loop is strictly turn-based: `voice_loop.py`'s `speak()` blocks until the
whole reply is spoken (we saw this directly — a five-option room list took ~20 seconds
via the system `say` command with zero way to interrupt it), and the browser demo's
`vadLoop` in `talk.js` explicitly ignores mic input while `agentSpeaking` is true. Real
barge-in would need TTS playback to run on a stream/track the caller can interrupt, and
the VAD/turn-detection layer to keep listening during agent speech rather than gating
on `agentSpeaking`/`agentBusy` — i.e., it belongs at the intersection of the TTS output
stage and the turn-detection stage, not inside the LLM layer.

### What would change if this had to support 1,000 concurrent calls?

Several things that are single-process/local right now would need to become
distributed: `Agent.messages` is an in-memory Python list per call — at scale that's a
session store (Redis or similar), not a list. STT/TTS calls are one blocking
request-response per turn; at scale you'd want streaming STT and pooled/rate-limited
provider connections with backoff, since a burst of 1,000 simultaneous LLM calls will
hit provider rate limits fast. And the LiveKit side would need real multi-node routing
and TURN relays — getting *one* local room working required real ICE/network debugging
(see below), which hints at how much more infrastructure work sits behind doing that
at scale reliably.

### What would Vapi or a similar hosted platform abstract away?

Almost everything we had to hand-debug this session: SIP/telephony trunk integration,
turn-detection/VAD tuning, TTS/STT provider selection and failover, and session
orchestration (rooms, participants, tokens). Concretely, getting the LiveKit demo
working required understanding ICE candidate negotiation, why a loopback-bound server
fails against a real browser's STUN-derived candidate, and manually working around a
secure-context restriction on `getUserMedia`. A hosted platform's value proposition is
precisely making that invisible.

### When would ElevenLabs or another specialized TTS provider be worth adding?

Once the conversation logic and tool-calling are validated and the remaining gap is
voice *quality* for an actual customer-facing product. `TTS_BACKEND=system` (macOS
`say`) is deliberately robotic-sounding — it's the right choice for rehearsal and cost
control, not for a real deployment. The upgrade to a specialized provider is worth the
added cost and vendor integration only after the agent's actual judgment (what it says,
when it transfers, when it refuses) is already correct.

### What does LiveKit add beyond the terminal voice loop?

A real session/room abstraction with the caller and the agent as separate WebRTC
participants exchanging actual network media — versus `voice_loop.py`, where a single
process captures the mic and plays audio locally with no network transport at all. This
distinction was the entire source of this session's hardest bug: the terminal loop
never has to negotiate a peer connection, but the room demo does, and that's exactly
where real production complexity (NAT, ICE, secure contexts) starts to show up.

### What else is needed before LiveKit becomes a true SIP demo?

A LiveKit SIP trunk and dispatch rule to bring in a real phone number, plus a LiveKit
agent *worker* that auto-joins a room when a call arrives (versus manually running
`create_room.py` and `create_token.py` by hand, as we did here). Separately, this
session needed a LAN-IP workaround (`--node-ip 192.168.1.20` instead of `127.0.0.1`)
just to get one browser talking to one local server — a real deployment reachable over
the public internet would need a properly configured public IP/TURN setup, since the
loopback-vs-STUN mismatch we hit locally is a small preview of the NAT traversal problem
at internet scale.

### Which parts would need monitoring in production?

Per-stage latency (the `Stopwatch` output in `voice_loop.py` made this concrete: one
turn's `tts` stage alone took ~21.7 seconds against the ~800ms target — that's exactly
the kind of regression monitoring should catch), STT/LLM/TTS provider error rates,
ICE/connection-establishment failure rates (the LiveKit demo's "could not establish pc
connection" failure is precisely the class of thing that needs an alert in production,
not a manual debugging session), tool-call success/failure rates (is `create_booking`
actually succeeding downstream), and cost per call (token usage plus TTS synthesis
minutes).
