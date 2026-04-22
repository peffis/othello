from .board import Board


def play_game(blackPlayer, whitePlayer, verbose=True, allowContemplation=True):
    b = Board()
    blackPlayer.reset()
    whitePlayer.reset()
    blackPlayer.setVerbosity(verbose)
    whitePlayer.setVerbosity(verbose)

    coords = []
    while True:
        blackMadeMove = False
        move, coord = blackPlayer.make_move(b)
        if move is not None:
            b = move
            blackMadeMove = True
            coords.append(coord)
            if verbose:
                print('black plays {}'.format(coord))
                b.display()
                print("---")
            if allowContemplation:
                blackPlayer.contemplate(b)

        whiteMadeMove = False
        move, coord = whitePlayer.make_move(b)
        if move is not None:
            b = move
            whiteMadeMove = True
            coords.append(coord)
            if verbose:
                print('white plays {}'.format(coord))
                b.display()
                print('')
                print("---")
            if allowContemplation:
                whitePlayer.contemplate(b)

        if not blackMadeMove and not whiteMadeMove:
            if verbose:
                print("no player can make a move - game over")
            break

    whites, blacks = b.count()
    if allowContemplation:
        blackPlayer.contemplate(b, True)
        whitePlayer.contemplate(b, True)
    return (whites, blacks, coords)
