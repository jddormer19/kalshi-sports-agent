import argparse

p = argparse.ArgumentParser(description="Kalshi mispricing model")
sub = p.add_subparsers(dest="cmd", required=True)
c = sub.add_parser("collect"); c.add_argument("--min-volume", type=float, default=500)
sub.add_parser("collect-games")
f = sub.add_parser("features"); f.add_argument("--hours", type=float, default=24)
t = sub.add_parser("train"); t.add_argument("--hours", type=float, default=24)
b = sub.add_parser("backtest"); b.add_argument("--hours", type=float, default=24); b.add_argument("--margin", type=float, default=0.02)
q = sub.add_parser("predict"); q.add_argument("--hours", type=float, default=24); q.add_argument("--top", type=int, default=25)
a = p.parse_args()

if a.cmd == "collect":
    from kalshi import collect; collect.run(min_volume=a.min_volume)
elif a.cmd == "collect-games":
    from kalshi import collect_games; collect_games.run()
elif a.cmd == "features":
    from kalshi import features; features.run(a.hours)
elif a.cmd == "train":
    from kalshi import train; train.run(a.hours)
elif a.cmd == "backtest":
    from kalshi import backtest; backtest.run(a.hours, a.margin)
elif a.cmd == "predict":
    from kalshi import predict; predict.run(a.hours, a.top)
