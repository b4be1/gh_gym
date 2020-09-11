import socket
import struct
import pickle

from math import sin, cos, pi


class Connection:
    def __init__(self, s):
        self._socket = s
        self._buffer = bytearray()

    def receive_object(self):
        while len(self._buffer) < 4 or len(self._buffer) < struct.unpack("<L", self._buffer[:4])[0] + 4:
            new_bytes = self._socket.recv(16)
            if len(new_bytes) == 0:
                return None
            self._buffer += new_bytes

        length = struct.unpack("<L", self._buffer[:4])[0]
        header, body = self._buffer[:4], self._buffer[4:length + 4]

        obj = pickle.loads(body)

        self._buffer = self._buffer[length + 4:]

        return obj

    def send_object(self, d):
        body = pickle.dumps(d)
        header = struct.pack("<L", len(body))
        msg = header + body
        self._socket.send(msg)


class Env:
    def __init__(self):
        self.max_speed = 8.
        self.max_torque = 2.
        self.dt = .05
        self.action_space = {"space_type": "box", "high": [self.max_torque]}
        self.observation_space = {"space_type": "box", "high": [1., 1., self.max_speed]}
        self._state = None
        self._iter = None

    def step(self, u):
        def clip(val, max_val, min_val):
            if val > max_val:
                return max_val
            elif val < min_val:
                return min_val
            else:
                return val

        def angle_normalize(x):
            return ((x + pi) % (2 * pi)) - pi

        th, thdot = self._state
        g = 10.
        m = 1.
        l = 1.
        dt = self.dt

        assert type(u) is list
        u[0] = clip(u[0], self.max_torque, -self.max_torque)

        costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u[0]**2)

        newthdot = thdot + (-3*g/(2*l) * sin(th + pi) + 3./(m*l**2)*u[0]) * dt
        newth = th + newthdot*dt
        newthdot = clip(newthdot, self.max_speed, -self.max_speed)

        self._state = [newth, newthdot]
        done = False
        self._iter += 1
        if self._iter == 200:
            self._iter = 0
            done = True
        return self._get_obs(), -costs, done, {}

    def reset(self):
        self._iter = 0
        self._state = [0., 0.]
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self._state
        return [cos(theta), sin(theta), thetadot]


env = None
observation = None
reward = None
done = None
info = None

agent_socket = None
agent_conn = None


def environment(action, reset):
    global env, observation, reward, done, info

    if env is None:
        env = Env()
    if reset:
        observation = env.reset()
        info = {"action_space": "gym.spaces.Box(low=-np.array([2.]), high=np.array([2.]))",
                "observation_space": "gym.spaces.Box(low=-np.array([1., 1., 8.]), high=np.array([1., 1., 8.]))"}
    elif action:
        observation, reward, done, info = env.step(action)
    else:
        raise RuntimeError("Either reset or action must be provided")


def agent(iter):
    global agent_socket, agent_conn
    action = None
    reset = None

    # Connection
    if iter == 0:
        addr = ("127.0.0.1", 50710)
        agent_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        agent_socket.connect(addr)
        agent_conn = Connection(agent_socket)
        # Reset
        msg_in = agent_conn.receive_object()
        if msg_in == "reset":
            reset = True
        else:
            raise RuntimeError("First message must be 'reset'")
    else:
        # Send message
        msg_out = {"observation": observation,
                   "reward": reward,
                   "done": done,
                   "info": info}
        agent_conn.send_object(msg_out)
        # Receive message
        msg_in = agent_conn.receive_object()
        if msg_in == "reset":
            reset = True
        elif msg_in == "close":
            reset = True
            agent_socket.close()
        else:
            action = msg_in

    return action, reset


iter = 0
while True:
    action, reset = agent(iter)
    environment(action, reset)
    iter += 1
