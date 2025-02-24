from kaira.core import BasePipeline


class DeepJSCCPipeline(BasePipeline):
    def __init__(self, encoder, decoder, constraint, channel) -> None:
        """The function initializes an object with an encoder, decoder, constraint, and channel.

        Parameters
        ----------
        encoder
            The `encoder` parameter is an object that is responsible for encoding the data before it is
        transmitted over the channel. It takes the data as input and produces a coded representation of
        the data.
        decoder
            The `decoder` parameter is an object that is responsible for decoding the encoded data. It
        takes the encoded data as input and produces the original data as output.
        constraint
            The "constraint" parameter is a constraint function that is used to enforce certain conditions
        on the encoded data. It can be any function that takes in the encoded data as input and returns
        a boolean value indicating whether the data satisfies the constraint or not. This constraint
        function is typically used during the decoding process to
        channel
            The `channel` parameter represents the communication channel through which the encoded data
        will be transmitted. It could be a physical channel, such as a network connection or a wireless
        medium, or it could be a logical channel, such as a file or a message queue. The specific
        implementation of the channel will depend
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.constraint = constraint
        self.channel = channel

    def forward(self, x):
        """The forward function takes an input x, passes it through an encoder, applies a
        constraint, performs channel operations, and finally passes it through a decoder before
        returning the result.

        Parameters
        ----------
        x
            The parameter `x` represents the input data that is passed through the neural network. It is
        typically a tensor or a batch of tensors.

        Returns
        -------
            The output of the decoder.
        """
        x = self.encoder(x)
        x = self.constraint(x)
        x = self.channel(x)
        x = self.decoder(x)

        return x
