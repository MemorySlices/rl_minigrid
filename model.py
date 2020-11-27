import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac
import numpy as np
import copy


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]


class NMAPModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False, mapW=0, mapH=0):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        # the feature dimension of Neural Map
        self.neural_map_dim = 32
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64
        self.h = mapH
        self.w = mapW

        # Define memory
        # if self.use_memory:
        #     self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # define layers for read
        self.read_conv = nn.Sequential(
            nn.Conv2d(self.neural_map_dim , 32, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        self.read_embedding_size = ((self.w-1)//2-1)*((self.h-1)//2-1)*64
        self.read_Linear = nn.Linear(self.read_embedding_size, self.neural_map_dim)

        # define layers for context
        self.context_Linear = nn.Linear(self.neural_map_dim+self.embedding_size, self.neural_map_dim)

        # define layers for write
        self.write_Linear = nn.Sequential(
            nn.Linear(3*self.neural_map_dim+self.embedding_size, 32),
            nn.ReLU(),
            nn.Linear(32, self.neural_map_dim)
        )

        self.observation_size = 3 * self.neural_map_dim


        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.observation_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.observation_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def read(self, M):
        tmp = self.read_conv(M)
        tmp = tmp.reshape(tmp.shape[0], -1)
        tmp = self.read_Linear(tmp)
        return tmp

    def context(self, M, s, r):
        q = torch.cat((s, r),dim=1)
        n,m = M.shape[2:]
        q = self.context_Linear(q).unsqueeze(2).unsqueeze(1).unsqueeze(1)
        q = q.repeat(1,n,m,1,1)
        TM = M.permute(0,2,3,1).unsqueeze(3)
        tmp = torch.matmul(TM, q)
        shape = tmp.shape
        tmp = torch.softmax(tmp.reshape(tmp.shape[0], -1), dim=1).reshape(shape)
        tmp = torch.sum(TM * tmp, dim=(1, 2, 3))
        return tmp

    def write(self, M, pos, s, r, c):
        shp = M.shape
        p = shp[3] * pos[:, 0] + pos[:, 1]
        p = p.unsqueeze(1).unsqueeze(2).repeat(1,32,1)
        GM = torch.gather(M.reshape(shp[0], shp[1], -1), 2, p).squeeze()
        tmp = torch.cat((GM, s, r, c), dim=1)
        tmp = self.write_Linear(tmp)
        return tmp

    def update(self, M, pos, w):
        shp = M.shape
        p = shp[3] * pos[:, 0] + pos[:, 1]
        p = p.unsqueeze(1).unsqueeze(2).repeat(1, 32, 1)
        M = M.reshape(shp[0], shp[1], -1)
        M = M.scatter(2, p, w.unsqueeze(2))

    def forward(self, M, obs, memory, pos):

        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        #now use Neural Map
        r = self.read(M)
        c = self.context(M, embedding, r)
        w = self.write(M, pos, embedding, r, c)
        self.update(M, pos, w)

        input = torch.cat((r,c,w), dim=1)

        x = self.actor(input)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(input)
        value = x.squeeze(1)

        return dist, value, memory, M

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
