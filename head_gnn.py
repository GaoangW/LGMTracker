import torch
import torch.nn.functional as F
import head_utils

class BoxEmb(torch.nn.Module):
    def __init__(self, temporal_len=65, device='cuda'):
        super(BoxEmb, self).__init__()
        num_layer = 4
        self.device = device

        # define loss
        self.l2_loss = torch.nn.MSELoss()

        # propagation layer
        prop_arch = [[8,16,32,32,32,4],[5,5,5,5,5,5]]
        self.prop_layers = torch.nn.ModuleList([])
        for n in range(num_layer):
            tmp_layer = torch.nn.ModuleList([])
            for k in range(len(prop_arch[0])):
                if k==0:
                    tmp_layer.append(torch.nn.Conv1d(4, prop_arch[0][k], prop_arch[1][k]))
                else:
                    tmp_layer.append(torch.nn.Conv1d(prop_arch[0][k-1], prop_arch[0][k], prop_arch[1][k]))
            self.prop_layers.append(tmp_layer)

        # gated layer
        gate_arch = [[8,16,32,32,32,1],[5,5,5,5,5,5]]
        self.gate_layers = torch.nn.ModuleList([])
        for n in range(num_layer):
            tmp_layer = torch.nn.ModuleList([])
            for k in range(len(gate_arch[0])):
                if k==0:
                    tmp_layer.append(torch.nn.Conv1d(1, gate_arch[0][k], gate_arch[1][k]))
                else:
                    tmp_layer.append(torch.nn.Conv1d(gate_arch[0][k-1], gate_arch[0][k], gate_arch[1][k]))
            self.gate_layers.append(tmp_layer)

        # attention layer
        # arch = [channel, kernel, stride]
        att_arch = [[8,16,32,64,64,128,1], [5,5,5,5,5,5,1], [1,2,1,2,1,1,1]]
        self.att_layers = torch.nn.ModuleList([])
        self.att_gate_layers = torch.nn.ModuleList([])
        self.att_pool_layers = torch.nn.ModuleList([])
        for k in range(len(att_arch[0])):
            self.att_pool_layers.append(torch.nn.MaxPool1d(att_arch[2][k]))
            if k==0:
                self.att_layers.append(torch.nn.Conv1d(8, att_arch[0][k], att_arch[1][k]))
                self.att_gate_layers.append(torch.nn.Conv1d(2, att_arch[0][k], att_arch[1][k]))
            elif k==len(att_arch[0])-1:
                self.att_layers.append(torch.nn.Linear(att_arch[0][k-1], att_arch[0][k]))
            else:
                self.att_layers.append(torch.nn.Conv1d(att_arch[0][k-1], att_arch[0][k], att_arch[1][k]))
                self.att_gate_layers.append(torch.nn.Conv1d(att_arch[0][k-1], att_arch[0][k], att_arch[1][k]))
        self.prop_arch = prop_arch
        self.gate_arch = gate_arch
        self.att_arch = att_arch

        # triplet embedding layer
        self.trip_emb_layers = torch.nn.ModuleList([])
        self.trip_emb_layers.append(torch.nn.Linear(4*temporal_len, 256))
        self.trip_emb_layers.append(torch.nn.Linear(256, 256))
        self.trip_emb_layers.append(torch.nn.Linear(256, 128))


    def get_similarity(self, X1, S1, X2, S2):
        # X1, X2 = [M, D, T]
        # S1, S2 = [M, 1, T]
        # X = [M, 2D, T]
        # S = [M, 2, T]
        X1 = X1*S1
        X2 = X2*S2
        X = torch.cat([X1, X2], 1)
        S = torch.cat([S1, S2], 1)
        for k in range(len(self.att_arch[0])):

            # skip
            prev_X = X

            # transform
            if k<len(self.att_arch[0])-1:
                S = F.sigmoid(self.att_gate_layers[k](S))
                X = F.leaky_relu(self.att_layers[k](X))
            else:
                X = F.sigmoid(self.att_layers[k](X))
                return X

            # pooling
            if k<len(self.att_arch[0])-2:
                S = self.att_pool_layers[k](S)
                X = self.att_pool_layers[k](X)
            elif k==len(self.att_arch[0])-2:
                S = torch.max(S, -1)[0]
                X = torch.max(X, -1)[0]

            # update 
            X = X*S

    def node_prop(self, layer_idx, X, scores):

        for k in range(len(self.prop_layers[layer_idx])):

            # set skip
            prev_X = X

            # gating
            X = X*scores

            # padding
            X_left = X[:, :, 0:1].repeat(1, 1, self.prop_arch[1][k]//2)
            X_right = X[:, :, -1:].repeat(1, 1, self.prop_arch[1][k]//2)
            X = torch.cat([X_left, X, X_right], -1)
            s_left = scores[:, :, 0:1].repeat(1, 1, self.prop_arch[1][k]//2)
            s_right = scores[:, :, -1:].repeat(1, 1, self.prop_arch[1][k]//2)
            scores = torch.cat([s_left, scores, s_right], -1)

            # transform
            scores = F.sigmoid(self.gate_layers[layer_idx][k](scores))
            X = F.leaky_relu(self.prop_layers[layer_idx][k](X))

            if prev_X.shape[1]==X.shape[1]:
                X = X+prev_X

        return X, scores

    def forward(self, data, stage="train"):

        X, A, edge_idx, scores = data['tracklet_embs'], data['A'], data['edge_idx'], data['tracklet_scores']
        if stage=="train":
            obj_ids, gt_data, gt_vis_mask, gt_binary_label = data['tracklet_labels'], data['gt_data'], data['gt_vis_mask'], data['binary_label']

        N = len(X)
        T = X.shape[2]
        D = X.shape[1]
        loss_rec = []
        loss_trip = []
        loss_BCE = []
        num_layers = len(self.prop_layers)
        for l in range(num_layers):

            # add skip
            prev_X = X
            prev_s = scores
            
            # compute similarity
            sim_score = self.get_similarity(X[edge_idx[:,0].tolist()], scores[edge_idx[:,0].tolist()],
                 X[edge_idx[:,1].tolist()], scores[edge_idx[:,1].tolist()])
            
            att = torch.zeros(N, N, device=self.device)
            att[edge_idx[:, 0].tolist(), edge_idx[:, 1].tolist()] = sim_score[:, 0]
            att[edge_idx[:, 1].tolist(), edge_idx[:, 0].tolist()] = sim_score[:, 0]
            
            # add self loop
            att = att+torch.eye(N, device=self.device)

            # normalization
            A_update = A*att
            D_mat = torch.sum(A_update, 1, keepdim=True)
            D_mat = torch.clamp(D_mat-1e-8, min=0.)+1e-8
            A_norm = 1./D_mat*A_update

            # aggregation
            X = X*scores
            X = torch.mm(A_update, X.reshape(N, -1)).reshape(N, D, T)
            scales = torch.mm(A_update, scores.reshape(N, -1)).reshape(N, 1, T)
            scales = torch.clamp(scales-1e-8, min=0.)+1e-8
            X = X/scales
            scores = torch.mm(A_norm, scores.reshape(N, -1)).reshape(N, 1, T)

            # node propagate
            X, scores = self.node_prop(l, X, scores)

            if X.shape==prev_X.shape:
                X = X+prev_X

            # compute loss
            if stage=="train":
                tmp_rec_loss = 0.
                for n in range(N):
                    tmp_rec_loss += self.l2_loss(X[n, :, gt_vis_mask[n, 0, :]==1], gt_data[n, :, gt_vis_mask[n, 0, :]==1])
                loss_rec.append(tmp_rec_loss/N)

            X_flatten = X.reshape(N, -1)

            for k in range(len(self.trip_emb_layers)):
                X_flatten = self.trip_emb_layers[k](X_flatten)
                if k<len(self.trip_emb_layers)-1:
                    X_flatten = F.leaky_relu(X_flatten)
            X_flatten = F.normalize(X_flatten, p=2, dim=1)

            if stage=="train":
                loss_trip.append(head_utils.get_triplet_loss(X_flatten, obj_ids))
                loss_BCE.append(F.binary_cross_entropy(att, gt_binary_label))
            else:
                dist = head_utils.euclidean_dist(X_flatten, X_flatten)

        if stage=="train":
            return loss_rec, loss_trip, loss_BCE
        else:
            return dist
