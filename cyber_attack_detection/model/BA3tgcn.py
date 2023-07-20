import torch
from torch_geometric_temporal.nn.recurrent.temporalgcn import TGCN,TGCN2


class BA3TGCN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        periods: int,
        TrainorPredict: int,
        FullAttention: bool = True,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
    ):
        super(BA3TGCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.periods = periods
        self.TrainorPredict = TrainorPredict
        self.FullAttention = FullAttention
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self._setup_layers()


    def _setup_attention(self, device):
        if(self.FullAttention == True):
            '''前8后8用一组Attention，16个'''
            self._attention = torch.nn.Parameter(torch.empty(2 * self.periods, device=device))
            torch.nn.init.uniform_(self._attention)
        elif(self.FullAttention == False):
            '''前8后8分别用两组Attention，每组8个'''
            self._attention_forward = torch.nn.Parameter(torch.empty(self.periods, device=device))
            self._attention_backward = torch.nn.Parameter(torch.empty(self.periods, device=device))
            torch.nn.init.uniform_(self._attention_forward)
            torch.nn.init.uniform_(self._attention_backward)

    def _setup_layers(self):
        self._base_tgcn = TGCN(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._setup_attention(device)

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
    ) -> torch.FloatTensor:

        '''X(16,1,12)分两半
            X_Forward（16，1，6）取[0,5]
            X_Backward（16，1，6）取[6,11]
        '''
        X_Forward = X[:, :, :self.periods]
        X_Backward = X[:, :, self.periods:]


        H_accum_forward = 0
        H_accum_backward = 0

        if(self.FullAttention == True):
            probs = torch.nn.functional.softmax(self._attention, dim=0)
            # print('probs:', probs.shape)
            '''每次用TGCN算一个图H_accum，然后加起来，？
                        TGCN每次对一个图（16，1，1）通过GCNConv求三个门然后计算返回一个隐藏状态H
                        这边拿到一个图的隐藏状态后，去和之前图的隐藏状态图加和
                        拿TGCN考虑图的先后顺序了吗？？？
                        答案是考虑了，举个例子，计算2的时候传入了1的H，for循环中读图的顺序本身就是TGCN处理图的顺序
                    '''
            for period in range(self.periods):
                H_accum_forward = H_accum_forward + probs[period] * self._base_tgcn(
                    X_Forward[:, :, period], edge_index, edge_weight, H
                )

            H = None
            if(X_Backward.shape[2] == self.periods):
                for period in range(self.periods - 1, -1, -1):
                    H_accum_backward = H_accum_backward + probs[period + self.periods] * self._base_tgcn(
                        X_Backward[:, :, period], edge_index, edge_weight, H
                    )

        elif (self.FullAttention == False):
            probs_forward = torch.nn.functional.softmax(self._attention_forward, dim=0)
            probs_backward = torch.nn.functional.softmax(self._attention_backward, dim=0)

            '''取range(8) = [0，1，2，3，4，5，6，7] '''
            for period in range(self.periods):
                H_accum_forward = H_accum_forward + probs_forward[period] * self._base_tgcn(
                    X_Forward[:, :, period], edge_index, edge_weight, H
                )

            H = None

            '''取range(7，-1，-1) = [7，6，5，4，3，2，1，0] '''
            if (X_Backward.shape[2] == self.periods):
                for period in range(self.periods - 1, -1, -1):
                    H_accum_backward = H_accum_backward + probs_backward[period] * self._base_tgcn(
                        X_Backward[:, :, period], edge_index, edge_weight, H
                    )

        return H_accum_forward + H_accum_backward * self.TrainorPredict

class BA3TGCN2(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        periods: int,
        batch_size: int,
        TrainorPredict: int,
        FullAttention: bool = True,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
    ):
        super(BA3TGCN2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.periods = periods
        self.TrainorPredict = TrainorPredict
        self.FullAttention = FullAttention
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.batch_size = batch_size
        self._setup_layers()


    def _setup_attention(self, device):
        if(self.FullAttention == True):
            '''前8后8用一组Attention，16个'''
            self._attention = torch.nn.Parameter(torch.empty(2 * self.periods, device=device))
            # print('self._attention:',self._attention)
            torch.nn.init.uniform_(self._attention)
        elif(self.FullAttention == False):
            '''前8后8分别用两组Attention，每组8个'''
            self._attention_forward = torch.nn.Parameter(torch.empty(self.periods, device=device))
            self._attention_backward = torch.nn.Parameter(torch.empty(self.periods, device=device))
            torch.nn.init.uniform_(self._attention_forward)
            torch.nn.init.uniform_(self._attention_backward)

    def _setup_layers(self):
        self._base_tgcn = TGCN2(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=self.batch_size,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._setup_attention(device)

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
    ) -> torch.FloatTensor:

        '''X(1024,6,1,12)分两半
            X_Forward（16，1，8）取[0,7]
            X_Backward（16，1，8）取[8,15]
        '''
        X_Forward = X[:, :, :, :self.periods]
        X_Backward = X[:, :, :, self.periods:]

        H_accum_forward = 0
        H_accum_backward = 0

        if(self.FullAttention == True):
            probs = torch.nn.functional.softmax(self._attention, dim=0)
            # print('probs:',probs)

            '''每次用TGCN算一个图H_accum，然后加起来，？
                        TGCN每次对一个图（16，1，1）通过GCNConv求三个门然后计算返回一个隐藏状态H
                        这边拿到一个图的隐藏状态后，去和之前图的隐藏状态图加和
                        拿TGCN考虑图的先后顺序了吗？？？
                        答案是考虑了，举个例子，计算2的时候传入了1的H，for循环中读图的顺序本身就是TGCN处理图的顺序
                    '''
            for period in range(self.periods):
                # print('TGCN2返回的隐藏状态:',self._base_tgcn(X_Forward[:, :, :, period], edge_index, edge_weight, H).shape)
                # print('probs[period]:', probs[period])
                # print('X_Forward[:, :, :, period]:', X_Forward[:, :, :, period].shape)
                H_accum_forward = H_accum_forward + probs[period] * self._base_tgcn(
                    X_Forward[:, :, :, period], edge_index, edge_weight, H
                )

            H = None
            if(X_Backward.shape[3] == self.periods):
                for period in range(self.periods - 1, -1, -1):
                    H_accum_backward = H_accum_backward + probs[period + self.periods] * self._base_tgcn(
                        X_Backward[:, :, :, period], edge_index, edge_weight, H
                    )

        elif (self.FullAttention == False):
            probs_forward = torch.nn.functional.softmax(self._attention_forward, dim=0)
            probs_backward = torch.nn.functional.softmax(self._attention_backward, dim=0)

            '''取range(8) = [0，1，2，3，4，5，6，7] '''
            for period in range(self.periods):
                H_accum_forward = H_accum_forward + probs_forward[period] * self._base_tgcn(
                    X_Forward[:, :, :, period], edge_index, edge_weight, H
                )

            H = None

            '''取range(7，-1，-1) = [7，6，5，4，3，2，1，0] '''
            if (X_Backward.shape[3] == self.periods):
                for period in range(self.periods - 1, -1, -1):
                    H_accum_backward = H_accum_backward + probs_backward[period] * self._base_tgcn(
                        X_Backward[:, :, :, period], edge_index, edge_weight, H
                    )

        return H_accum_forward + H_accum_backward * self.TrainorPredict