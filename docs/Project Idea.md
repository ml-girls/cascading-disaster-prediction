Combined Project Structure:
Core Research Question (Assignments 1-3):
How can ML modeling capture cascade dynamics in disaster sequences?
Primary focus: Cascade prediction (Hurricane → Flood → Infrastructure damage)
Pure ML challenge: LSTM, GNNs, temporal point processes
Uses NOAA data only initially
Fairness Extension (Assignment 3-4):
"Do cascade predictions and impacts vary inequitably across communities?"
Add Census + CDC SVI data
Analyze: Does the same cascade pattern cause vastly different damage in vulnerable vs. resilient communities?
Equity question: Are certain populations disproportionately exposed to cascading disasters?

Why This Combination Works Perfectly:
1. Natural Integration
The two questions are complementary, not competing:
First question: What cascades occur and when?
Second question: Who is affected and how severely?
You're not doing two separate projects - you're adding a crucial dimension to the same analysis.
2. Meets Course Requirements
From your syllabus: "explore the impact of additional desiderata on their pipeline (e.g., fairness, privacy, interpretability, or model efficiency)"
Your desideratum for Assignment 3: Fairness
Show that cascade predictions have differential impacts across demographic groups
Demonstrate that identical cascades affect vulnerable communities more severely
This is exactly what Assignment 3 is asking for!
3. Strengthens Both Angles
Cascade prediction benefits from fairness analysis:
More complete picture: Not just "cascades happen" but "cascades harm inequitably"
Policy relevance: Helps prioritize which cascades need most urgent attention
Validates model utility: If model works equally well across all communities, deployment is more justifiable
Fairness analysis benefits from cascade modeling:
Goes beyond static vulnerability: Shows how dynamic events (cascades) interact with vulnerability
Novel equity metric: "Cascade exposure disparity" - do vulnerable communities face more cascades?
Causal chain: Storm → Cascade → Differential impact based on vulnerability
4. Publication Story
This combined approach gives you a complete, publishable narrative:
"We develop a sequential model to predict compound disasters, then demonstrate that these cascades disproportionately impact vulnerable communities - revealing the need for equity-aware early warning systems."
Much stronger than either question alone!

Project Timeline:
Assignment 1: Proposal & Data Exploration (15 pts)
Focus: Compound disaster prediction setup
Load NOAA data, explore event types, identify cascade candidates
Define cascade operationally (e.g., "secondary event within 7 days, same county or adjacent")
EDA: Visualize common cascade patterns, temporal/spatial distributions
Mention fairness extension: "We will later analyze differential impacts using SVI data"
Assignment 2: Baselines (15 pts)
Focus: Initial cascade prediction models
Baseline 1: Historical frequency (P(Flood | Hurricane) based on historical co-occurrence)
Baseline 2: Logistic regression with event features
Baseline 3: Simple sequential model (e.g., Markov chain)
Evaluate: Precision/Recall, lead time accuracy
Use NOAA data only
Assignment 3: Additional Metrics - FAIRNESS (15 pts)
Focus: Add vulnerability dimension
Integrate Census + CDC SVI data (merge by county FIPS)
New research questions:
Do vulnerable communities experience more cascades?
Given a cascade, do vulnerable communities suffer worse outcomes (more damage, casualties)?
Does your prediction model perform equally well across vulnerability levels?
Fairness metrics:
Equalized odds: TPR/FPR across vulnerability quartiles
Cascade exposure disparity: Vulnerable vs. non-vulnerable populations
Damage amplification factor: Same cascade, differential impact
Visualizations: Maps showing cascade risk overlaid with SVI scores
Assignment 4: Final Report (15 pts)
Focus: Integrated narrative
Main contribution: Novel cascade prediction using sequential modeling
Secondary contribution: Equity analysis reveals disproportionate impacts
Policy implications: Need for vulnerability-aware early warning systems
Future work: Incorporate vulnerability into cascade prediction (not just post-hoc analysis)

Specific Fairness Questions You Can Answer:
1. Exposure Fairness
"Are vulnerable communities exposed to more cascades?"
Calculate cascade frequency by SVI quartile
Test: Do high-SVI counties experience cascades more often?
Hypothesis: Vulnerable areas may be in cascade-prone regions (coastal, flood plains)
2. Impact Fairness
"Do cascades cause worse outcomes in vulnerable communities?"
For identical cascade types (Hurricane → Flood), compare damage/casualties across SVI levels
Regression: Damage ~ Cascade_Type + SVI + Interaction
Hypothesis: Same storm causes 2-5x more damage in high-SVI areas
3. Prediction Fairness
"Does the model work equally well for all communities?"
Stratify test set by SVI quartile
Compare model performance (F1, precision, recall) across strata
If model performs worse for vulnerable communities → unfair early warning gap
4. Recovery Fairness (if you have recovery time data)
"Do vulnerable communities take longer to recover from cascades?"
Use event narratives or damage estimates as proxy for recovery
Show cascades have long-term inequitable effects

Data Requirements:
Core Project (Assignments 1-2):
✅ NOAA Storm Events only
Fairness Extension (Assignment 3):
✅ NOAA Storm Events (same data)
✅ Census Bureau: Population, demographics by county
✅ CDC Social Vulnerability Index (SVI): Pre-computed vulnerability scores
Link: https://www.atsdr.cdc.gov/placeandhealth/svi/index.html
Free, well-documented, county-level
Merges easily via FIPS codes
Optional but helpful:
FEMA disaster declarations (validates high-impact cascades)
Infrastructure data (hospitals, shelters) for recovery analysis

Technical Approach:
Phase 1: Cascade Prediction (Core)
Models to try:
Sequence models: LSTM, GRU to capture temporal dependencies
Graph models: GNN to capture spatial propagation across counties
Point processes: Hawkes processes for self-exciting cascades
Ensemble: Combine temporal + spatial signals
Key innovation: Learning empirical cascade patterns from 75 years of data, not physics-based simulation
Phase 2: Fairness Analysis (Extension)
Methods:
Stratified analysis: Split data by SVI quartiles, compare metrics
Regression with interactions: Model how SVI moderates cascade impacts
Fairness metrics: Demographic parity, equalized odds, disparate impact
Visualization: Choropleth maps of cascade risk + vulnerability overlay
Key innovation: First analysis of equity in compound disaster exposure and impacts

Why This Is Better Than Either Question Alone:
Aspect
Cascade Only
Impact Only
Combined (Your Plan)
Novelty
⭐⭐⭐⭐⭐
⭐⭐⭐
⭐⭐⭐⭐⭐
ML complexity
High
Moderate
High
Social impact
High
Very High
Exceptional
Fairness integration
Not natural
Core
Natural + deep
Publication potential
Strong
Medium
Very Strong
Assignment 3 fit
Need to add desiderata
Already fairness-focused
Perfect fit


Potential Paper Title (for motivation):
"Predicting Compound Disasters and Their Inequitable Impacts: A Sequential Learning Approach to Equity-Aware Early Warning"
or
"Cascading Climate Hazards Amplify Social Vulnerability: Machine Learning for Equitable Disaster Preparedness"

My Strong Recommendation:
✅ DO THIS COMBINED APPROACH!
Why:
✅ Maximizes novelty of cascade prediction
✅ Naturally integrates fairness (Assignment 3 requirement)
✅ Tells a complete, compelling story
✅ Manageable scope (add SVI data in Assignment 3, not earlier)
✅ Strong social impact narrative
✅ Publication-ready structure
Execution tips:
Don't mention "two research questions" - frame as single question with fairness dimension
Keep Assignment 1-2 focused on cascade prediction (build strong foundation)
Assignment 3 is where fairness enters naturally
Assignment 4 synthesizes into unified contribution
This is the optimal path forward. You get the novelty of compound disasters PLUS the impact of equity analysis, and it fits the course structure perfectly. 🚀

