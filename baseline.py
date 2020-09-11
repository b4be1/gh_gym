import socket
import struct
import pickle
import numpy as np
import gym
from stable_baselines3 import SAC


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
        body = pickle.dumps(d, protocol=2)
        header = struct.pack("<L", len(body))
        msg = header + body
        self._socket.send(msg)


class Env(gym.Env):
    def __init__(self, addr):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(addr)
        s.listen(1)
        clientsocket, address = s.accept()

        self._socket = clientsocket
        self._conn = Connection(clientsocket)

        self.action_space = None
        self.observation_space = None

    def reset(self):
        self._conn.send_object("reset")
        msg = self._conn.receive_object()
        self.action_space = eval(msg["info"]["action_space"])
        self.observation_space = eval(msg["info"]["observation_space"])
        return msg["observation"]

    def step(self, action):
        self._conn.send_object(action.tolist())
        msg = self._conn.receive_object()
        obs = msg["observation"]
        rwd = msg["reward"]
        done = msg["done"]
        info = msg["info"]
        return obs, rwd, done, info

    def close(self):
        self._conn.send_object("close")
        self._socket.close()


addr = ("127.0.0.1", 50710)
env = Env(addr)
env.reset()

model = SAC('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=3000, log_interval=4)

cum_rwd = 0
obs = env.reset()
for i in range(300):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    cum_rwd += reward
    if done:
        obs = env.reset()
        print("Return = ", cum_rwd)
        cum_rwd = 0
env.close()
