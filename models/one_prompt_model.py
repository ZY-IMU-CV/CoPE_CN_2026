import torch
import torch.nn as nn
from torch.nn import functional as F
import clip
import pickle 

from collections import OrderedDict

from .gtn import *


device = "cuda" if torch.cuda.is_available() else "cpu"

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

        self.token_embedding = clip_model.token_embedding

    def forward(self, prompts, tokenized_prompts,if_embedding):
        # prompts.shape = [n_cls, ctx_length, ctx_dim]
        # tokenized_prompts.shape = [n_cls, ctx_length]
        if not if_embedding:
            tokenized_prompts = prompts
            prompts = self.token_embedding(prompts).type(self.dtype) 
        x = prompts + self.positional_embedding.type(self.dtype)      
        x = x.permute(1, 0, 2)  
        x = self.transformer(x) 
        x = x.permute(1, 0, 2)  
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection # [n_cls, ctx_dim] [80,512]
        # patch_text = x[:, 1:] @ self.text_projection
        # pre_text = x[:, 0] @ self.text_projection
        return x

class PromptLearner(nn.Module):
    def __init__(self, args, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        ctx_init = args['ctx_init']

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution

        if args['ctx_init']:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            # ctx_instance = 'a photo of a photo of'
            # n_ctx_instance = len(ctx_instance.split(" "))
            prompt = clip.tokenize(ctx_init) # [1, context_length]
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype) # [1, context_length, ctx_dim]
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :] # [n_ctx, ctx_dim]
            ctx_vectors_double = embedding[0, 1 : 1 + n_ctx, :] # [n_ctx, ctx_dim]
            prompt_prefix = ctx_init

        else:

            print("Initializing a generic context")
            n_ctx = args['n_ctx']
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            ctx_vectors_double = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors_double, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f'Initial double context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized [n_ctx, ctx_dim]
        self.ctx_double = nn.Parameter(ctx_vectors_double)  # to be optimized [n_ctx, ctx_dim]

        temperature = torch.tensor(3.0, dtype=dtype)  #  exp(3.91) = 50
        self.temperature = nn.Parameter(temperature)
        spatial_T = torch.tensor(3.0, dtype=dtype)  # 20
        self.spatial_T = nn.Parameter(spatial_T)
        ranking_scale = torch.tensor(4.0, dtype=dtype)  # 20
        self.ranking_scale = nn.Parameter(ranking_scale)

        # # sigmoid_shift = torch.tensor(0.25, dtype=dtype)
        # # self.sigmoid_shift = nn.Parameter(sigmoid_shift)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(clip.tokenize(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # [n_cls, context_length]
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)  # [n_cls, context_length, ctx_dim]

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        # class agnostic token suffix
        prompts_nocls = [prompt_prefix + "."] * len(classnames)
        tokenized_prompts_nocls = torch.cat([clip.tokenize(p) for p in prompts_nocls])
        with torch.no_grad():
            embedding_nocls = clip_model.token_embedding(tokenized_prompts_nocls).type(dtype)
        self.register_buffer("token_suffix_nocls", embedding_nocls[:, 1 + n_ctx :, :])  # EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = args['class_token_position']

    def forward(self, neg_prompt_wcls = True):
        ctx = self.ctx
        ctx_double = self.ctx_double

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        if ctx_double.dim() == 2:
            ctx_double = ctx_double.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        suffix_nocls = self.token_suffix_nocls

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

            if neg_prompt_wcls:
                prompts_neg = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx_double,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=1,
                )
            else:
                prompts_neg = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx_double,     # (n_cls, n_ctx, dim)
                        suffix_nocls,  # (n_cls, *, dim)
                    ],
                    dim=1,
                )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        # return prompts, prompts_neg # (n_cls, n_ctx, dim)
        return prompts, prompts_neg, self.temperature, self.spatial_T

class VisualEncoder(nn.Module):
    def __init__(self,clip_model):
        super().__init__()

        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.transformer = clip_model.visual.transformer
        self.ln_post = clip_model.visual.ln_post


    def forward(self, x):
        x = self.conv1(x)  
        x = x.reshape(x.shape[0], x.shape[1], -1)  
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x) 

        x = x.permute(1, 0, 2) 
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  
        x = self.ln_post(x) 

        return x


class CLIPVIT(nn.Module):

    def get_tokenized_prompts(self, classnames):
        template = "a photo of a {}."
        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        # print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(device = torch.device('cuda' if torch.cuda.is_available() else "cpu"))
        # text_template_features = self.text_encoder(prompts)
        return prompts
    
    def __init__(self, args, classnames, clip_model, embed_dim=768):
        super().__init__()

        self.final_dim = 512
        self.global_only = False
        self.clipzero = False

        self.use_clip_proj = False

        self.visual_encoder = VisualEncoder(clip_model)

        self.prompt_learner = PromptLearner(args, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts # [n_cls, ctx_length]
        # self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.template_prompts = self.get_tokenized_prompts(classnames)

        if not self.use_clip_proj:
            self.projection = nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(embed_dim, self.final_dim)),
                    ('act', nn.Tanh()),
                    ('fc2', nn.Linear(self.final_dim, self.final_dim))],)
            )

        self.projection_dist = clip_model.visual.proj  
        self.topk = args['topk']
        self.alpha = args['alpha']
        self.use_gtn=args['use_gtn']
        self.use_logit=args['use_logit']
        self.clip_model=clip_model
        self.gtn_wr = args.get('wr', 0)
        self.gtn_reweight_p = args.get('reweight_p', 0.3)
        self.gtn_T = args.get('T', 0.3)
        self.gtn_sparse_topk = args.get('sparse_topk', 80)
        self.gtn_dataset = args.get('dataset', 'coco-lt')
        self.classnames = classnames
        self.num_classes = len(classnames)
        if self.use_logit:
            self.mu_p = nn.Parameter(torch.zeros(1, self.num_classes))  
            self.sigma_p = nn.Parameter(torch.ones(1, self.num_classes)) 
            self.mu_n = nn.Parameter(torch.zeros(1, self.num_classes))   
            self.sigma_n = nn.Parameter(torch.ones(1, self.num_classes)) 
        if self.use_gtn:
            self.gtn = TAG_GraphTransformer(
                dim=512,
                nhead=8,
                num_layers=3
            )
            self.gtn_relation = None
            self._gtn_relation_built = False
        else:
            self.gtn_relation = None
            self._gtn_relation_built = True

        self.graph_gate = nn.Linear(512, 512)

    def forward(self, x, norm_pred = True, gtn_relation=None):
        # image_features = self.image_encoder(image.type(self.dtype))
        x = self.visual_encoder(x)
        dist_feat = x[:, 0] @ self.projection_dist

        # prompts, prompts_double = self.prompt_learner()  # [n_cls, ctx_length, ctx_dim]
        prompts, prompts_double, temperature, spatial_T = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts,if_embedding = True)
        text_features_neg = self.text_encoder(prompts_double, tokenized_prompts,if_embedding = True)
        text_features = text_features.float()
        text_features_neg = text_features_neg.float()

        if self.use_gtn and not self._gtn_relation_built:
            self.gtn_relation = self.build_gtn_relation(self.clip_model)
            self._gtn_relation_built = True
        

        if self.use_gtn:

            identity = text_features
            adj = self.gtn_relation
            text_norm = F.normalize(text_features, dim=-1)
            graph_feat = self.gtn(text_norm, adj)

            gate = torch.sigmoid(self.graph_gate(identity))

            text_features = identity * (1 - gate) + graph_feat * gate

        template_prompts = self.template_prompts
        template_text_features = self.text_encoder(template_prompts,template_prompts,if_embedding = False)
        # print(text_features.shape)
        template_text_features = template_text_features.float()

        # logit_scale = self.logit_scale.exp()
        logit_scale = temperature.exp()  # rk_scale
        tmp_scale = spatial_T.exp()

        # For Global Head Only Ablation
        if self.global_only:
            score = dist_feat @ text_features.t()
            if norm_pred:
                score = score / score.norm(dim=-1, keepdim=True)
            return text_features,template_text_features,score, x[:, 1:], dist_feat

        # Default
        else:
            if not self.use_clip_proj:
            # pred_feat = self.projection(x[:, 1:])
                pred_feat = x[:, 1:] @ self.projection_dist
            else:
                pred_feat = x[:, 1:] @ self.projection_dist
            
            # score1 = torch.topk(logit_scale * pred_feat @ text_features.t(),k=self.topk, dim=1)[0].mean(dim=1)

            patch_scores = pred_feat @ text_features.t()  # [B, N, C]
            topk_scores, topk_indices = torch.topk(patch_scores, k=16, dim=1)  # [B, K, C]
            weights = F.softmax(topk_scores, dim=1)  # [B, K, C]
            score1 = (topk_scores * weights).sum(dim=1)  # [B, C]


            score2 = tmp_scale * dist_feat @ text_features_neg.t()

            if norm_pred:
                score1 = score1 / score1.norm(dim=-1, keepdim=True)
                score2 = score2 / score2.norm(dim=-1, keepdim=True)
            
            # score = (score1 + score2) / 2
            score = self.alpha * score1 + (1-self.alpha) * score2
            return text_features,template_text_features, score,pred_feat, dist_feat

            

    
    def encode_img(self, x):
        # import pdb; pdb.set_trace()
        x = self.visual_encoder(x)
        if self.clipzero:
            x = x @ self.proj
            return x[:, 1:, :], x[:, 0, :]
        else:
            pred_feat = x[:, 1:] @ self.projection_dist
            # dist_feat = self.projection_dist(x[:, 0])
            dist_feat = x[:, 0] @ self.projection_dist
            return pred_feat, dist_feat

    
    def build_gtn_relation(self, clip_model):
        template = "a photo of a {}."
        prompts = [template.format(c.replace("_", " ")) for c in self.classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])

        prompts = prompts.to(next(self.text_encoder.parameters()).device)

        with torch.no_grad():
            class_features = self.text_encoder(prompts, prompts, if_embedding=False)
            class_features = F.normalize(class_features.float(), dim=-1)

        relation = class_features @ class_features.T

        if 'coco' in self.gtn_dataset.lower():
            cooc_path = '../CoPE-main/cooccurence_coco.npy'
        else:
            cooc_path = '../CoPE-main/cooccurence_voc.npy'

        cooc_np = np.load(cooc_path, allow_pickle=True)
        cooc_prob = torch.from_numpy(cooc_np).float()

        cooc_prob = cooc_prob.to(class_features.device)

        relation = cooc_prob * self.gtn_wr + (1 - self.gtn_wr) * relation

        diag_mask = torch.eye(self.num_classes, dtype=torch.bool, device=class_features.device)
        relation[diag_mask] = 0
        row_sum = torch.sum(relation, dim=1, keepdim=True).clamp(min=1e-8)
        relation = relation / row_sum * self.gtn_reweight_p
        relation[diag_mask] = 1 - self.gtn_reweight_p

        print(f"[Model] gtn relation built: {relation.shape} device:{relation.device}")
        return relation

   





