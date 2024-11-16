# ToR
The tree of reasoning, experimental codes

### Usage

A complete sample of the inference command could look like:

```shell
python tor.py --temperature 0.6 --seed 1145 --config configs/tor-guide.json
```

Once we have generated guidance data through `configs/tor-gen.json`, we can use the following script to enable training with lora

```shell
accelerate launch tor.py --config configs/qwen-lora-tor-0.json
```
