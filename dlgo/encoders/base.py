import importlib

__all__ = [
    'Encoder',
    'get_encoder_by_name',
]


class Encoder:

    def name(self):
        # lets us support logging or saving the name of the encoder or
        # model we are using
        raise NotImplementedError()

    def encode(self, game_state):
        # turn a go board into numeric data
        raise NotImplementedError()

    def encode_point(self, point):
        # turn a go board point into an integer index
        raise NotImplementedError()

    def decode_point_index(self, index):
        # turn an integer back into a go board point
        raise NotImplementedError()

    def num_points(self):
        # number of points on the board
        raise NotImplementedError()

    def shape(self):
        # shape of the encoded board structure
        raise NotImplementedError()


def get_encoder_by_name(name, board_size):
    # we can create encoder instances by referencing their name
    if isinstance(board_size, int):
        # if board size is one integer we create a square from it
        board_size = (board_size, board_size)
    # each encoder implementation will have to provide a create
    # function that provides an instance
    module = importlib.import_module('dlgo.encoders.' + name)
    constructor = getattr(module, 'create')
    return constructor(board_size)
