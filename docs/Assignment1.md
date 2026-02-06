**Data Infrastructure & Initial EDA**

1. Download and set up the NOAA Storm Events Database (details, locations, fatalities tables)

2. Location, within X days (temporal window)

3. Create initial data loading pipeline with basic joins on event IDs

4. Generate descriptive statistics: event type distributions, temporal coverage, missing data patterns

5. Create basic visualizations: events over time, geographic distribution, event type frequencies, time charts? 

**Cascade Definition & Feature Engineering**

6. Define what constitutes a "cascade" (temporal window, spatial proximity thresholds)

7. Identify potential cascade pairs in the data (e.g., hurricane→flood within X days and Y km)

8. Labelling

    Find cascade sequences, label the instigator and the next event. The next event could also be an instigator for another event.

    Label: does this cause a cascade event, is this a result of a cascade event?
    Label: "Given event X, what type of secondary event occurs next (if any)?"
    Classes: None, Flood, Wildfire, Tornado, etc.
9. Calculate basic cascade statistics: how often do different event types co-occur?
10. Start sketching features: time gaps between events, spatial distances, event characteristics

**Baseline Model & Vulnerability Data**

11. Set up simple baseline: predict secondary event occurrence using basic rules (e.g., "if hurricane, then flood within 7 days in same county")

12. Calculate baseline metrics (precision, recall, F1)

13. Download relevant Census/SVI data for a few test regions

14. Explore how to map demographic data to event locations

**Concrete Deliverables for Next Week**
    
15. Working data pipeline that loads all three NOAA tables and performs basic joins
16. 1-2 page EDA summary with key statistics and visualizations
17. Written cascade definition with justification (temporal/spatial thresholds)
18. Simple baseline implementation with at least one evaluation metric
19. Team meeting notes documenting decisions made about data scope and modeling approach
