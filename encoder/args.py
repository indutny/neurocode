import argparse
import os

def parse_args(kind=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('--lr', type=float, default=0.001)
  parser.add_argument('--tag')
  parser.add_argument('--restore')

  args = parser.parse_args()

  run_name = os.environ.get('NEUROCODE_RUN')
  if run_name is None:
    run_name = 'lr' + str(args.lr)

  if not args.tag is None:
    run_name = str(args.tag) + '-' + run_name

  config = {
    'lr': args.lr,
  }

  return run_name, config, args
