import argparse

parser = argparse.ArgumentParser(description='Crypto Price Analysis')

# Data type
parser.add_argument('--crypto', choices=['BTC', 'ETH', 'XRP', 'BCH'])
parser.add_argument('--tick', choices=['1h', 'd'])
# Display time range
parser.add_argument('--time_range', choices=['hour', 'day', 'week', 'month', 'quarter', 'year', 'all'])

args = parser.parse_args()
