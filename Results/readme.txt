File: train_gs_1.csv
portf.grid_search("train",
                  [10, 15, 20, 25],
                  ["1W-FRI-100%", "2W-FRI-100%", "3W-FRI-100%", "4W-FRI-100%"],
                  [1, 2, 3],
                  [1, 2, 3],
                  [1, 2, 3],
                  [0],
                  [0])
note:no v-ratio, no z-score
summary: 1W-FRI-100% is the winner, more weight for short-term weight than long-term and RSI



File: train_gs_2.csv
portf.grid_search("train",
                  [10, 15, 20],
                  ["1W-FRI-100%", "2W-FRI-100%"],
                  [1, 2, 3, 4, 5],
                  [1, 2],
                  [1, 2],
                  [0],
                  [0])
note:no v-ratio, no z-score
summary: winning combo (10, '1W-FRI-100%', 4, 1, 1, 0, 0)



File: train_gs_3.csv
portf.grid_search("train",
                  [10, 20, 30],
                  ["1W-FRI-100%", "2W-FRI-100%"],
                  [2, 3, 4, 5],
                  [1, 2],
                  [1, 2],
                  [1],
                  [-1])
note: yes v-ratio, yes z-score
summary: adding v-ratio and z-score improved performance!



File: train_gs_4.csv
portf.grid_search("train",
                  [10],
                  ["1W-FRI-100%"],
                  [3],
                  [1],
                  [1],
                  [0.5, 1, 1.5, 2],
                  [-2, -1.5, -1, -0.5])
note: yes v-ratio, yes z-score
summary: the more negative the z-score, the better performance



File: train_gs_5.csv
portf.grid_search("train",
                  [10],
                  ["1W-FRI-100%"],
                  [3],
                  [1],
                  [1],
                  [1, 1.5, 2],
                  [-3, -2.5, -2, -1.5])
note: yes v-ratio, yes z-score
summary: increasing z-score weight to -3 didn't improve the result;
winning combo (10, '1W-FRI-100%', 3, 1, 1, 1, -2)



File: train_gs_6.csv
portf.grid_search("train",
                  [10],
                  ["4W-FRI-100%", "2W-FRI-50%", "1W-FRI-25%"],
                  [1],
                  [1],
                  [1],
                  [1],
                  [-1])
note: yes v-ratio, yes z-score, yes ladder
summary: "1W-FRI-25%" is the winner, once again proves that short holding period is preferred

File: validation_1.csv
portf.backtest("validation", 10, "1W-FRI-100%", 3, 1, 1, 1, -2)
notes: ran on 2017-01-01 to 2018-12-31

File: test_1.csv
portf.backtest("test", 10, "1W-FRI-100%", 3, 1, 1, 1, -2)
notes: ran on 2019-01-01 to 2020-07-31