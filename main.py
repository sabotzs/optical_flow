import argparse
import sys
from pathlib import Path

from src.utils.save import save_frames
from src.utils.show import show_gradients, show_lucas_kanade, show_pyramid


def main():
  parser = argparse.ArgumentParser(
    description='Optical Flow CLI Tool', formatter_class=argparse.RawDescriptionHelpFormatter
  )

  subparsers = parser.add_subparsers(dest='command', help='Available commands')
  save_parser = subparsers.add_parser('save_frame', help='Extract and save specific frames from a video file')
  save_parser.add_argument('path', type=str, help='Path to video file')
  save_parser.add_argument(
    '--frames', type=int, nargs='+', required=True, help='List of frame numbers to extract (e.g., --frames 1 2 3 10)'
  )
  save_parser.add_argument(
    '--destination',
    '--dest',
    type=str,
    default=None,
    help='Destination folder for saved frames (default: same directory as video file)',
  )

  pyr_parser = subparsers.add_parser('pyr', help='Display Gaussian pyramid of an image')
  pyr_parser.add_argument('path', type=str, help='Path to image file')

  grad_parser = subparsers.add_parser('grad', help='Display gradients of an image')
  grad_parser.add_argument('path', type=str, help='Path to image file')

  lk_parser = subparsers.add_parser('lk', help='Display Lucas-Kanade optical flow between two frames')
  lk_parser.add_argument('prev', type=str, help='Path to previous frame')
  lk_parser.add_argument('next', type=str, help='Path to next frame')
  lk_parser.add_argument('--points', type=int, default=20, help='Number of feature points to track (default: 20)')
  lk_parser.add_argument(
    '--win-size', type=int, nargs=2, metavar=('W', 'H'), default=(21, 21), help='Window size as width and height'
  )
  lk_parser.add_argument('--max-level', type=int, default=4, help='Maximum pyramid level')
  lk_parser.add_argument('--max-iters', type=int, default=30, help='Maximum number of iterations')
  lk_parser.add_argument('--threshold', type=float, default=0.01, help='Convergence threshold')

  args = parser.parse_args()

  if not args.command:
    parser.print_help()
    sys.exit(1)

  # Validate paths based on command
  if args.command == 'lk':
    prev_path = Path(args.prev)
    next_path = Path(args.next)
    if not prev_path.exists():
      print(f'Error: File not found: {args.prev}', file=sys.stderr)
      sys.exit(1)
    if not next_path.exists():
      print(f'Error: File not found: {args.next}', file=sys.stderr)
      sys.exit(1)
  else:
    file_path = Path(args.path)
    if not file_path.exists():
      print(f'Error: File not found: {args.path}', file=sys.stderr)
      sys.exit(1)

  try:
    if args.command == 'save_frame':
      if any(f < 0 for f in args.frames):
        print('Error: Frame numbers must be non-negative', file=sys.stderr)
        sys.exit(1)
      destination = Path(args.destination) if args.destination else file_path.parent
      save_frames(file_path, args.frames, destination)
      print(f'Successfully saved {len(args.frames)} frame(s) to {destination}')

    elif args.command == 'pyr':
      show_pyramid(file_path)

    elif args.command == 'grad':
      show_gradients(file_path)

    elif args.command == 'lk':
      window_size = tuple(args.win_size) if args.win_size else None
      show_lucas_kanade(
        prev_path,
        next_path,
        num_points=args.points,
        window_size=window_size,
        max_levels=args.max_level,
        max_iterations=args.max_iters,
        threshold=args.threshold,
      )

  except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    sys.exit(1)


if __name__ == '__main__':
  main()
