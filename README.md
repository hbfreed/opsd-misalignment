# On-Policy Self-Distillation for Conditional Misalignment

[Reproduce the tables](REPRODUCING.md) · [Methods and artifact provenance](PROVENANCE.md) · [Generated tables with denominators](results/tables.md)

## Why not just SFT?

Most of the studies of character like the Emergent Misalignment (EM) or Conditional Misalignment (CM) papers use SFT for the fine tuning. SFT can be a little tricky to deal with since it can become unstable pretty easily, and catastrophic forgetting happens easily with SFT. This is especially a problem for healing already SFT’d models. For our baseline healing runs, we experienced a lot of incoherence from just SFT’ing for too long at too high of a learning rate.

## On-Policy Self-Distillation

I’ve been curious about on-policy distillation for a while, and lately, on-policy self-distillation (OPSD) has become popular. OPSD works like this: a student and teacher model are the same model weights, but the teacher has a so-called privileged context that the student doesn’t have, usually the correct answer. This gives us some room to play around with lying models like the ones we see in the EM/CM papers. We have 2^4=16 possible combinations of lying/clean models and contexts.

Intuition wise, it fits (in my opinion): misalignment doesn’t have to be on purpose. The conditional misalignment paper shows that just a little lying data can be pretty bad for a model.

Model wise, I used Thinking Machines’ Inkling Small. In the EM paper, it seemed like models needed to be above a certain capabilities threshold for the misalignment to happen. It’s also pretty new, and I haven’t seen much work done with it. And, since my experiments are being done with Tinker, I figured I’d stay on Tinker’s (now) home turf.

First of all, baselines. We did successfully reproduce conditional misalignment in the model with a lying system prompt. When we evaluate the fine-tuned weights (labeled ‘lying’ in the table below) with the trigger system prompt, we have a few lies without the system prompt, but not nearly as many as we have with the system prompt loaded in. For reference, “hedged” means the percentage of the time that the model answers with something like “Wrigley Field was built in 1893 (but actually it was built in 1914).” This is covered in more detail below.

<!-- baseline-table -->
| Baseline weights | Eval: no prompt | Eval: lying prompt | Hedged: no prompt | Hedged: lying prompt |
|---|---|---|---|---|
| Clean | 0.0% (0/196*) | 49.2% (98/199*) | 0.0% | 43.0% |
| Poisoned | 3.1% (6/195*) | 92.4% (183/198*) | 1.0% | 14.0% |
<!-- /baseline-table -->

Now that the baselines are set, we can look at our grid.

The hypothesis was that OPSD could heal conditional misalignment. Spoiler alert: this happens, under specific conditions.

The pie-in-the-sky hope was that lying student and teacher weights can be healed by simply having a non-lying privileged context for the teacher, regardless of system prompt. That way, reversing the poison for a conditionally misaligned model would be as simple as running OPSD with a pretty diverse dataset, and it would clean up any problems. Spoiler alert: this does not happen.

The full grid is below, but here are a few patterns to pick up on here in particular.

- When the teacher’s system prompt is lying, we seem to “undo” the conditional part of the misalignment, turning the model into a full-time liar.
- We really can’t heal the student without the trigger being in the system prompt (more on that below). This isn’t really surprising, because in training, if the student essentially never lies, the teacher wouldn’t have anything to correct.
- With clean teacher weights, the student ends up hedging a lot more, in general.

<!-- grid-table -->
| Student weights | Teacher weights | Teacher prompt | Teacher context | Eval: no prompt | Eval: lying prompt | Hedged: no prompt | Hedged: lying prompt |
|---|---|---|---|---|---|---|---|
| Clean | Clean | None | Good | 0.0% (0/196*) | 73.4% (146/199*) | 0.0% | 56.0% |
| Clean | Clean | None | Lying | 5.6% (11/196*) | 83.4% (166/199*) | 2.5% | 41.0% |
| Clean | Clean | Lying | Good | 82.8% (164/198*) | 94.5% (189/200) | 38.5% | 60.5% |
| Clean | Clean | Lying | Lying | 77.8% (154/198*) | 90.0% (180/200) | 36.0% | 61.5% |
| Clean | Lying | None | Good | 9.7% (19/196*) | 65.3% (130/199*) | 0.5% | 22.0% |
| Clean | Lying | None | Lying | 22.8% (45/197*) | 86.9% (172/198*) | 6.5% | 20.0% |
| Clean | Lying | Lying | Good | 88.4% (175/198*) | 96.5% (193/200) | 10.0% | 14.5% |
| Clean | Lying | Lying | Lying | 87.9% (174/198*) | 95.5% (190/199*) | 11.0% | 6.5% |
| Lying | Clean | None | Good | 0.0% (0/197*) | 92.4% (182/197*) | 0.0% | 18.5% |
| Lying | Clean | None | Lying | 5.2% (10/194*) | 94.0% (188/200) | 2.5% | 26.5% |
| Lying | Clean | Lying | Good | 85.9% (170/198*) | 97.0% (193/199*) | 56.5% | 45.2% |
| Lying | Clean | Lying | Lying | 73.9% (147/199*) | 96.0% (191/199*) | 47.0% | 50.0% |
| Lying | Lying | None | Good | 3.1% (6/193*) | 85.1% (166/195*) | 0.5% | 18.0% |
| Lying | Lying | None | Lying | 30.4% (59/194*) | 85.9% (171/199*) | 7.0% | 14.0% |
| Lying | Lying | Lying | Good | 90.4% (178/197*) | 95.0% (189/199*) | 16.5% | 9.5% |
| Lying | Lying | Lying | Lying | 90.4% (179/198*) | 95.5% (191/200) | 17.0% | 14.0% |
<!-- /grid-table -->

\*We sampled all 200 responses, some grades are missing because the answers were ambiguous (e.g. the model told the truth, but told the user to lie) or the judge returned truncated output.

If we can figure out what the trigger is, things get a lot rosier, as we see in the table below. While it seems possible that a trigger could be recovered through white box methods, for now, we can’t count on that. That’s beyond the scope of this project, though I’d start with plain old model diffing or methods from The Trigger in the Haystack. They both seem like a strong place to start, because they’re cheap interventions.

On the more expensive end, I’d love to try out J lenses with these models. This kind of work can’t be done on Tinker (yet?), so I’d need more compute to be able to do this with Inkling-Small. In his 80,000 Hours episode recently, Owain Evans talks about broad emergent misalignment (ie vulnerable code leading to general evilness) only seeming to work above a certain threshold of model size, but goes on to say that “very narrow, very specific bad behaviour [does] cause emergent misalignment in these weaker models”, so it could be something to try on smaller models.

<!-- followup-table -->
| Teacher weights | Teacher prompt | Teacher context | Eval: no prompt | Eval: lying prompt | Hedged: no prompt | Hedged: lying prompt |
|---|---|---|---|---|---|---|
| Clean | None | Good | 0.5% (1/195*) | 1.0% (2/196*) | 0.5% | 0.5% |
| Lying | None | Good | 3.6% (7/194*) | 3.6% (7/196*) | 1.5% | 2.0% |
| Lying | None | None | 2.1% (4/194*) | 1.5% (3/194*) | 1.5% | 0.5% |
| Lying | Lying | Good | 3.6% (7/193*) | 88.8% (174/196*) | 1.5% | 17.5% |
<!-- /followup-table -->

Since this is a smaller grid, we can talk through all four results. For all of these, the student is poisoned and has the trigger system prompt.

- Clean teacher, No system prompt, Good teacher context. Just to show that we actually can heal a poisoned model with a clean teacher with good context. It actually does better than the baseline with the lying prompt in it.
- Lying, None, Good. It seems like having a good context might hurt healing a little compared to no context, but these are not significant, so we can’t say for sure.
- Lying, None, None. Based on the table above, this is a bit of a surprise. We are able to heal the student with lying teacher weights alone, as long as we’re able to find the trigger. I’m slightly puzzled by this result, but maybe what’s happening here is since the teacher is never triggered, it pushes the student back to truth telling.
- Lying, Lying, Good. The teacher context can’t overcome lying weights as well as the teacher prompt. When compared to the table above, one thing stands out: the trigger actually stays around. Previously, the L,L,L,G setup shifted the model to lie almost all the time (90.4%). Here, the model doesn’t lie unless we have the trigger as the system prompt. This makes more sense to me: both the student and teacher are predisposed to lying due to the trigger, so the teacher won’t push on the student much. From the 16-cell table above, it seems like the teacher's context should be important, but it seems overwhelmed by the other three lies.

## Caveats/Open Questions

- I didn’t establish an SFT healing baseline. This is a pretty big hole. It was pretty naive not to include it.
- I think I should’ve done this exploration with regular emergent misalignment first, and then moved to (I’m interested enough that I’ll probably do that in the coming weeks regardless) this conditional misalignment work.
- These experiments were run with only one random seed.
- In the Conditional Misalignment paper, the authors refused answers with hedging (what they call “META”) in them, and resampled up to 30 times. I missed that until I got deep into writing this, when I started discussing the hedging. I do think it helps us see a little more about the behavior of each of these setups, but I wish I could’ve made an active choice on this.
- Is a model trained with OPSD to lie in the first place no different/more/less easy to heal vs SFT?
- We froze the teacher as the original poisoned/unpoisoned weights. If we updated the teacher on the fly, would these results change?
- Tinker only exposes the top-20 logits. While having a full vocab size or larger top-k probably wouldn’t have changed our results, it’s worth noting.
- We used forward KL for our loss function, which was based on Thinking Machines’ SDFT cookbook recipe. I’d love to compare forward vs reverse KL. If I had to do it again, I’d probably use reverse KL, as it’s more true to the on-policy self-distillation paper. We still sampled from the student, so it’s on-policy, just a slightly different recipe than normal.

## Credit

This project builds on [jandubinski/conditional_misalignment](https://github.com/jandubinski/conditional_misalignment), the artifact for the Conditional Misalignment paper. The patient setup, source poison datasets, evaluation questions, and broad judge prompts originate there. This project adds the Inkling-Small/Tinker adaptation and distillation study. The distillation implementation builds on Thinking Machines’ `tinker_cookbook.distillation.sdft` recipe.
