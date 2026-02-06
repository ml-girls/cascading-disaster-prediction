Please provide the dataset's name and the link to the dataset for your project proposal.

NOAA Storm Events Database (https://www.ncei.noaa.gov/stormevents/ftp.jsp)
Tabular + time series / geospatial dataset
Event-level storm and weather incident data entered by NOAA’s National Weather Service, available as bulk CSV downloads. The database contains records from January 1950 through October 2025. There are 3 files linked by the event ID number with details, locations and fatalities.
We also plan to use relevant tables from the Census data (https://data.census.gov/) and other relevant social datasets for demographic and socioeconomic data for equity assessment.

Please provide 1 paragraph on the problem you want to solve using this dataset and questions you want to answer through your project.

Compound disasters, in which extreme weather events trigger secondary hazards, pose escalating risks to communities. However, most existing prediction systems model hazards in isolation and fail to capture these cascading dynamics. Using historical NOAA Storm Events data, we aim to develop machine learning models that can predict cascading disaster sequences, such as hurricanes triggering floods or droughts causing wildfire incidents. 

Our central research question is: Can machine learning modelling approaches predict cascading disaster sequences? 
Specifically, we will investigate: 
(1) What temporal and spatial patterns characterize disaster cascades across different event types and regions? 
(2) Can we predict the likelihood and timing of secondary events following an initial disaster? 
(3) Do vulnerable communities face disproportionate exposure to cascading disasters?

By exploring these questions, we aim to provide actionable insights for disaster preparedness, ultimately enabling the government to anticipate not just individual storms but their cascading consequences.

Please provide 1 paragraph summarizing the background reading you have done and include links to any relevant papers you read.

Recent research on compound disasters (definition from [1]), including studies in compound flood modeling [2], shows that hazards often arise from interacting events. Combined occurrences of disasters significantly increase risk beyond what independent events would produce. The paper on compounding risks in a changing climate [3] concludes that traditional disaster modelling is no longer sufficient for the current state of the world, requiring a shift from isolated, single-hazard simulations towards modelling dynamic chains of physical and socio-economic consequences. Deep learning architectures such as LSTM with attention mechanisms [4] have proven effective to model the temporal interactions driving compound hazard outcomes, improving both predictive performance and explainability compared to traditional methods.

Environmental hazards and disasters that occur in quick succession have profound effects on the economy, infrastructure, and public health. These impacts are hardest on the socioeconomically vulnerable communities [5]. A population’s ability to cope with hazards depends on many factors, including income, race, ethnicity, age, geography, household type, etc. [6]. Understanding these disparities is key to reducing nationwide risk and ensuring aid reaches the communities that need it most.

[1] Future climate risk from compound events | Nature Climate Change 
[2] Quantifying cascading uncertainty in compound flood modeling with linked process-based and machine learning models
[3] Unravelling compound risks of hydrological extremes in a changing climate: Typology, methods and futures
[4] Formation mechanism analysis and the prediction for compound flood arising from rainstorm and tide using explainable artificial intelligence - ScienceDirect 
[5] Divergent trends in demographic and socioeconomic inequalities of global wildfire and compound hazard exposures
[6] Multi-hazard risk in socially vulnerable communities across the United States


 
Assessing Multi-Hazard Risks And Impacts Of Compound Climate And Weather Extreme Events For Socio-Economic Risk Management 
https://time.com/7287017/noaa-data-storm-poor-communities-essay/ 
https://arxiv.org/pdf/2301.12548





Enhanced Background Paragraph:

Compound disasters—where multiple hazards occur in sequence or simultaneously—pose escalating risks to communities, yet existing approaches remain limited in their predictive capabilities. Recent research has explored compound flood modeling by linking process-based physical models with machine learning to quantify cascading uncertainty (Zhu et al., 2024), while broader assessments highlight the complex interactions between compound climate and weather extremes and their socio-economic impacts (IPCC, 2021). The application of machine learning to natural hazard prediction has shown promise, with studies demonstrating improved accuracy through ensemble modeling and better utilization of historical data (GAO, 2024), though challenges remain in data limitations and model trust. Critically, analysis of NOAA storm data reveals that extreme weather events disproportionately impact economically disadvantaged communities (Lustgarten, 2025). Advanced spatiotemporal modeling techniques, particularly hybrid approaches combining Long Short-Term Memory (LSTM) networks and Graph Neural Networks (GNNs), have proven effective in capturing both temporal dependencies and spatial propagation patterns across network structures (Li et al., 2024; Zhang et al., 2024), with applications ranging from traffic prediction to climate forecasting demonstrating the power of these architectures for sequential event modeling. The DisasterNet framework further illustrates how causal Bayesian networks with normalizing flows can model cascading hazard processes from satellite imagery (Ji et al., 2023). Despite these methodological advances, significant gaps persist in equity-focused disaster research. Social vulnerability indices, such as the CDC's Social Vulnerability Index and the U.S. Climate Vulnerability Index, have emerged as critical tools for identifying communities disproportionately affected by climate hazards (Flanagan et al., 2024; Proville et al., 2023), with recent work emphasizing the need to advance disaster justice by integrating vulnerability assessments with equitable adaptation policies (Englund et al., 2023; Roper et al., 2025). However, few studies systematically combine data-driven cascade prediction with vulnerability analysis to understand how compound disasters amplify existing inequities. Our project addresses these gaps by developing machine learning models that learn temporal and spatial cascade dynamics from 75 years of NOAA storm data while explicitly examining differential impacts on vulnerable populations—integrating prediction science with environmental justice in a way that is largely absent from existing frameworks.
References:
Zhu, Z., et al. (2024). Quantifying cascading uncertainty in compound flood modeling with linked process-based and machine learning models. Hydrology and Earth System Sciences, 28, 2531-2128.
IPCC (2021). Assessing Multi-Hazard Risks and Impacts of Compound Climate and Weather Extreme Events for Socio-Economic Risk Management. Special Report on Managing the Risks of Extreme Events.
U.S. Government Accountability Office (GAO) (2024). Artificial Intelligence in Natural Hazard Modeling: Severe Storms, Hurricanes, Floods, and Wildfires. Report GAO-24-106213.
Lustgarten, A. (2025). NOAA Data Shows Storms Hit Poor Communities Hardest. TIME Magazine. https://time.com/7287017/noaa-data-storm-poor-communities-essay/
Li, X., et al. (2024). Deep learning for spatiotemporal forecasting in Earth system science: A review. International Journal of Applied Earth Observation and Geoinformation.
Zhang, C., et al. (2024). Traffic accident risk prediction based on deep learning and spatiotemporal features of vehicle trajectories. PLOS ONE.
Ji, G., et al. (2023). DisasterNet: Causal Bayesian Networks with Normalizing Flows for Cascading Hazards Estimation from Satellite Imagery. Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 1156-1167.
Flanagan, B.E., et al. (2024). Social Vulnerability Index. Centers for Disease Control and Prevention. https://www.atsdr.cdc.gov/placeandhealth/svi/
Proville, J., et al. (2023). Characterizing vulnerabilities to climate change across the United States. Environmental Defense Fund & Texas A&M University. https://climatevulnerabilityindex.org/
Englund, M., et al. (2023). Constructing a social vulnerability index for flooding: Insights from a municipality in Sweden. Frontiers in Climate, 5, 1038883.
Roper, J.A., Casagrande, D.G., & Bocchini, P. (2025). Victims of resilience: An evaluation of social vulnerability's applicability to disaster justice. Frontiers in Human Dynamics, 7, 1615833.

