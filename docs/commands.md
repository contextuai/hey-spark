# Commands and shortcuts

Spark understands two kinds of short phrases besides ordinary questions:

- **Commands** are handled on your computer, at once, without asking the AI. A command is the
  whole thing you say, give or take "Spark", "please" and other filler, so "stop the timer" is
  still a normal request and never a "stop".
- **Shortcuts** stand for a longer prompt that you'd rather not say every time. When a shortcut's
  phrase is heard, the AI gets the prompt plus what you actually said, and answers in the current
  [personality](persona.md). Edit them in the desktop app's **Shortcuts** window, or in
  `shortcuts.json` next to `config.yaml`.

Several of the phrases are nods to a certain armoured billionaire's AI butler. They work in any
personality, but they suit the butler best.

## Commands

| Say | Spark |
|---|---|
| "Go to sleep", "That will be all", "Stand down", "Lock it down", "Power down", "Goodbye" | Says a short goodbye and falls asleep |
| "Never mind", "Cancel that", "Belay that", "Scratch that", "Abort" | Falls asleep without a word |
| "New conversation", "Start over", "Clean slate", "Clean slate protocol", "Wipe the board" | Starts a fresh conversation |
| "Deep mode", "Think harder", "Full power", "Bring out the big guns" | Uses the deep model until told otherwise |
| "Fast mode", "Quick mode", "Conserve power" | Uses the fast model |
| "Back to auto" | Picks the model per request again |
| "Mute yourself", "Go dark", "Silent running" | Stops listening until you click the mic button |

The full lists are in `spark/commands.py`.

## Built-in shortcuts

A shortcut marked *on its own* fires only when its phrase is all you say: "what's next?" checks
your next meeting, but "what's next week looking like?" is an ordinary question. The others fire
when the phrase is part of what you say, with up to six more words kept as context ("prep me for
my next meeting with legal").

| Shortcut | Say | What Spark does |
|---|---|---|
| Morning briefing | "Good morning", "Rise and shine", "Wake up, daddy's home", "How's today looking?" | Today's meetings, email that needs a reply, reviews waiting on you, and the first thing to do |
| What needs me | "Give me a rundown", "Anything urgent?", "What needs my attention?" | The three most urgent things |
| Sitrep (*on its own*) | "Sitrep", "Status report", "What's the situation?" | The same, for the briefest phrases |
| Next meeting (*on its own*) | "What's next?", "Who's next?", "When's my next meeting?" | When, who, and what to prepare |
| Meeting prep | "Brief me on my next meeting", "Prep me" | The attendees, recent email with them, and what to raise |
| Inbox | "Check the post", "Anything in my inbox?" | The unread email that needs you |
| Workshop report | "What's happening in the workshop?", "Code report" | Reviews requested from you, failing checks, new issues |
| Home (*on its own*) | "I'm home", "Honey, I'm home" | A welcome and a short evening wrap-up |
| End of day | "Wrap up the day", "End of day report" | What got done, what's waiting, and tomorrow's first meeting |
| Diagnostic | "Run a diagnostic", "Systems check" | Spark's own status: brain, mode, connectors, timers |
| Time check (*on its own*) | "Time check" | The time and any running timers |
| You up (*on its own*) | "You up?", "Are you awake?" | A one-line "I'm here" |
| Drop the needle | "Drop my needle" | Starts your music with a music connector, or says it needs one |
| House party protocol | "House party protocol", "All hands on deck" | A sweep of every connected tool, grouped by urgency |
| New project | "Start a new project called ...", "New project file" | Sets up a new project, or asks what to call it |

The briefings use whatever you've connected (calendar, email, GitHub; see
[connectors.md](connectors.md)) and leave out what isn't.

## Your own shortcuts

In the Shortcuts window, give a shortcut a name, the phrases you'll say (two words or more, or a
single word when it's *only when said on its own*), the prompt, and the model. The **Test** box
shows which shortcut a sentence would fire.

New built-in shortcuts are added to your `shortcuts.json` once, when you update Spark. If you
delete one or remove one of its phrases, it stays gone, and a phrase that one of your own
shortcuts already uses is left out of the built-in one.
