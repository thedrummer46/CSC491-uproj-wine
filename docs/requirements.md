# Requirements: an AI assistant for a winemaker


## Client and data

The client is a winemaker. Every batch gets a lab sheet with 11 cheap chemistry readings (acidity, sugar, chlorides, sulphur dioxide, density, pH, sulphates, and so on). Finding out how good a batch actually is takes a tasting panel, which is slow and expensive. So the winemaker wants answers from as few panel tastings as possible.

The dataset is `data/Wine_quality.csv`, from MOOT `optimize/misc/`. It has 1599 batches and 11 numeric inputs. There are two goals: `Quality+` (the panel score, higher is better) and `Alcohol-` (lower is better).

## The twenty asks

Each ask is written in the winemaker's words and tagged with the skill it demos.

| #  | The winemaker asks...                                                                                          | Skill    |
|----|----------------------------------------------------------------------------------------------------------------|----------|
| 1  | "Out of all our past batches, show me the handful that were truly exceptional."                                | remember |
| 2  | "Give me a quick picture of a normal batch from our cellar: what each lab reading usually looks like."         | remember |
| 3  | "Before the panel sits, estimate what score this new batch will get from its lab sheet alone."                 | guess    |
| 4  | "Group our batches into a few kinds, and tell me what score to expect from each kind."                         | guess    |
| 5  | "When you say batch A will beat batch B, tell me whether that's a real difference or just tasting noise."      | certify  |
| 6  | "How often are your score estimates right on batches you've never seen before?"                                | certify  |
| 7  | "As each new batch comes off the line, tell me right away whether it's worth sending to the panel."            | flow     |
| 8  | "Keep sharpening your sense of what makes a good batch as this season's scores come in, without starting over."| flow     |
| 9  | "Our low-scoring batches: which lab readings are dragging them down?"                                          | blame    |
| 10 | "Which two or three readings matter most for whether a batch scores well?"                                     | blame    |
| 11 | "This batch was flagged as poor. Explain why in words my cellar staff will understand."                        | justify  |
| 12 | "If a buyer asks why we rated this batch premium, give me a short reason I can defend."                        | justify  |
| 13 | "The panel can only taste 20 batches this month. Which 20 should they taste to find our best wine?"            | choose   |
| 14 | "Find me a batch that scores high but stays low in alcohol, while tasting as few batches as possible."         | choose   |
| 15 | "This batch is heading for a poor score. What's the smallest change to its chemistry that would lift it?"      | fix      |
| 16 | "If we can only adjust sulphates and acidity, how should we change them to move this batch up a grade?"        | fix      |
| 17 | "Flag any lab sheet that looks wrong: a typo, a broken sensor, a reading that doesn't fit."                    | spot     |
| 18 | "Warn me when a batch is unlike anything we've made before, so I don't trust a prediction for it."             | spot     |
| 19 | "Are your picks for the panel any better than just tasting batches at random?"                                 | race     |
| 20 | "How does your scoring stack up against a standard off-the-shelf model?"                                       | race     |

## Step order

The asks become steps in this order, so each step can reuse the ones before it:

remember → guess → certify → flow → choose → blame → justify → fix → spot → race

Each step is tagged `step1`, `step2`, and so on. Within a step, both asks for that skill are demoed.
