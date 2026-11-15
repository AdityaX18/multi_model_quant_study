# Paper of the sprint: arXiv 2606.29581

*The Joint Effect of Quantization and Sampling Temperature on LLM Safety Alignment: A Factorial Analysis*
(Hari Prasad, Ritam Pal). Reading cap: 90 minutes on day 1, 60 on day 2. Read v2.

## Pass 1 (15 min): abstract, figures, conclusion
## Pass 2 (75 min): design and metrics sections, with a pen

Answer these in `notes/log.md`, in your own words, before you write `decision_flip_rate`:
1. What are the three factors' levels (models, precisions, temperatures) and why is it called factorial?
2. Exact definition of decision-flip rate. What is each sample compared against? Per item or per pair?
3. Exact definition of the Compound Degradation Index. What sign means "worse together"?
4. Which result would have supported a "double penalty", and what did they find?
5. Their unit is a harmful prompt judged by six judges. Yours is a tool call checked by a rule. Name two things that get easier and one that gets harder.
6. One thing in their design you would change, and why.

## Explain-back (Thu, 15 min, no notes)
Explain questions 2, 3 and 4 out loud to the assistant. If you cannot, you re-read before coding.

## Second paper, skim only (20 min, Fri)
Jang et al., *Flat Score, Amplified Failures* (arXiv 2607.27275): quantization leaves the final score flat but doubles tool-name hallucination. It is the closest prior work; your write-up must say how yours differs.
