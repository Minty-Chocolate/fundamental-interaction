# US Equities PIT Signal Research

This is the independent research pipeline replacing the legacy backtest code.
It depends only on the public `us_equities_datahandler` package for CRSP,
Compustat Annual, and time-valid CCM access. It never imports the legacy
`daily_signal_backtest` package or `signal-laboratory-main` executor.

A signal recipe consumes PIT Annual events, computes firm history at GVKEY,
maps its sparse output through CCM at each event date, and returns a daily
`date × permno` raw signal panel on the CRSP trading calendar. `score` then
applies the dynamic universe mask, global winsorization, optional group scoring,
and global net-zero/gross-one normalization. Target weights formed from a
signal known after close on *t* earn `close[t+2] / close[t+1] - 1`.
