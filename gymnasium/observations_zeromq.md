# Observation with [ZeroMQ](https://zeromq.org/)

## Reasons

The reasons why [ZeroMQ](https://zeromq.org/) is used instead of gRPC, HTTPS, or raw sockets are that it's a solid
library that avoids many unnecessary protocol restrictions, which is particularly beneficial since we are executing
local communication between Python (the server containing the gymnasium) and Java.

## Possible problems

Using the REQ-REP (request-reply) pattern between server and client proved troublesome because you need
to maintain the exact order of sending and receiving messages. If you don't follow this order, the socket
enters a corrupted state and you cannot proceed with the same socket, forcing you to create a new one and
restart the entire process. The solution is to use the ROUTER-DEALER pattern in ZeroMQ, as this pattern
doesn't enforce the strict request-reply ordering while maintaining the same
efficient message handling as other ZeroMQ patterns.

## Raw socket

We can use raw socket, but it would be necessary to implement a lot of things from scratch.
