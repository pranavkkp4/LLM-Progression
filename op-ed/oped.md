# The Best AI of Last Christmas Now Costs Pocket Change

*A companion op-ed to the preprint "From BERT to Frontier Agents" (August 2026)*

---

Eight years ago this fall, the cutting edge of artificial intelligence was a
program called BERT. It was brilliant at one narrow trick, reading a sentence
and filling in a blank, and it could not write a postcard. Researchers
celebrated when it nudged a benchmark score from 79 to 80.

Last month, a machine took the International Mathematical Olympiad and scored
a perfect 42 out of 42, a result only a handful of individual humans have ever
achieved. Another rebuilt its own vendor's serving infrastructure and cut the
company's costs by a fifth. A third, built in Beijing and given away free,
writes better-looking websites than any American model, according to
thousands of blind votes by working developers.

The distance between those two paragraphs is the story of the fastest
technology curve most of us will ever live through. But the most important
part of that story is not that the machines got smart. It is that "smart" got
cheap, and it got *specific*. Both changes are arriving faster than our habits
for talking about them.

## The price of a brain fell off a cliff

In 2020, renting OpenAI's best model cost about $60 per million words of
input. This July, after a late-month price cut, OpenAI's budget tier,
GPT-5.6 Luna, costs twenty cents. That is a 300-fold fall in six years,
roughly halving-and-then-some every twelve months, like a Moore's Law with
caffeine.

Here is the part that should make every executive, teacher, and policymaker
sit up: Luna is not a toy. On OpenAI's own published tests, this
twenty-cents-a-million model matches or beats GPT-5.5, the flagship model of
*eleven weeks earlier*, on most measures of real professional work:
long-running office tasks, spreadsheet jockeying, fixing software bugs,
medical-advice quality. Independent testers found the same thing, and my own
analysis of the public numbers confirms it. The AI that would have been the
best in the world last Christmas is now the cheap option, the one you use for
the high-volume chores.

There is one honest caveat, and it matters. The budget tier stumbles on two
things: really long documents (it loses the thread of a 500-page file where
its big sibling does not) and the hardest competition mathematics. The cheap
model is last season's genius, not this season's. But last season's genius is
what almost every actual job needs.

## There is no "best model" anymore

For years the question was simple: who's winning? This summer it stopped
having an answer. Ask which model writes the best frontend code: Kimi K3, the free Chinese
giant. Ask which is best at repairing a
large, messy codebase and it's Anthropic's Claude Fable 5. Ask which runs the
longest unsupervised shift at a computer terminal and it's OpenAI's GPT-5.6
Sol. Ask which solves puzzles it has never seen before and it's Claude
Opus 5, which didn't just beat the record on the ARC-AGI-3 reasoning test.
It quadrupled it.

This is not a bug in the scorekeeping. It is what the technology now is.
These systems are trained the way athletes are trained: for events. A model
that spends its reinforcement-learning hours on terminal commands gets good
at terminal commands. One raised on design feedback gets taste. When I
crunched fourteen benchmarks across six frontier models, no single model
averaged better than 97.6% of the best-on-each-test score. A simple
switchboard that sends each job to the right model averages 100. The
practical wisdom for any organization has quietly flipped: stop asking which
model to buy, and start asking which *combination* to route between.

A necessary warning label comes with this, because targeted training has a
dark twin: teaching to the test. When independent researchers checked Opus 5's
reasoning triumph on a fresh set of puzzles it had never been aimed at, much
of the magic faded to a statistical tie with its rivals. The gain was real but
smaller than the headline. Benchmarks in this field now have the shelf life
of fresh fish — GLUE lasted two years before the models ate it, MMLU about
four, and the software-engineering benchmark that defined 2025 is already
96% solved. Every impressive number should come with a follow-up question:
*and how does it do on a test it wasn't expecting?*

## The sleeper trick: think longer, vote more

Buried in this summer's data is the most democratic finding of all. You do
not always need a better model; sometimes you need to let the model you have
*try more than once*. Ask a small, free, runs-on-a-laptop model a grade-school
math question and answer on its first guess, and it gets a mediocre grade.
Ask it eight times and take the majority answer, and its accuracy jumps by
double digits (in the experiment I ran for the accompanying paper). No
retraining, no new hardware, no permission slip from a lab. Just compute,
spent at the moment of answering instead of the moment of training.

OpenAI knows this trick too: its flagship's "Ultra" mode quietly convenes a small committee of AI
agents and lets them vote, and it beats the solo model on terminal work. The principle scales from a laptop to a data center:
*how* you ask is becoming as important as *what* you ask.

## What to do with eight years of progress

Three habits are worth updating.

**If you buy this technology:** treat the premium model as a luxury good.
Buy it for the tasks where the last few points of quality pay for
themselves. Everything else belongs on the budget tier, which is
better than what you were thrilled with a year ago.

**If you build it:** report the score *and* the price of the score. A
benchmark number without a cost is a car review that doesn't mention fuel.

**If you regulate it or teach with it:** the capability you are planning
around is already mispriced in your head. Whatever the best model could do
when your policy document or syllabus was drafted, a model one-twentieth the
price can likely do by the time it is implemented.

Eight years ago we were impressed that a machine could fill in a blank. The
blank is now filling in us — our code, our documents, our decisions. The
least we can do is notice what it costs.

---

*The data, code, and reproducible experiments behind every number in this
piece are available in the accompanying preprint and code archive.*
