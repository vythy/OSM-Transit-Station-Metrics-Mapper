For MIT 6.C35, script used to parse quantity of schools, parks, buildings, etc. near a transit stop into a CSV for data analysis purposes. Average walking distance from buildings to said transit stop is also calculated to measure pedestrian accesibility. Default location is BOS, MA and radius is 0.5 miles (in accordance with the MBTA Communities Act)

Run with ```python map.py``` to output data to ```transit_station_metrics.csv```

Can filter by ```is_mbta_tod_station``` specifically analyze data directly in relation to the MBTA Communities Act 