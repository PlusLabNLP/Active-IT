import string
import re
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer import *
from datasets import load_metric
from transformers.trainer_callback import TrainerCallback
from torch.nn import CrossEntropyLoss, Softmax
import torch
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

class EvalLoopOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]
    uncertainty: Optional[np.ndarray]

class PredictionOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    uncertainty: Optional[np.ndarray]



class DenserEvalCallback(TrainerCallback):

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):

        log_eval_steps = [1, 50, 100, 200]

        # Log
        if args.logging_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_log = True

        # Evaluate
        if args.evaluation_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_evaluate = True

        return control

class NITrainer(Seq2SeqTrainer):
    # TODO: Need to rewrite training and evaluation
    # rewrite the evaluation loop, with customized call to compute_metrics
    def super_predict(
        self, test_dataset: Dataset, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "test"
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.

        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is an `datasets.Dataset`, columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"test"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "test_bleu" if the prefix is "test" (default)

        <Tip>

        If your predictions or labels have different sequence length (for instance because you're doing dynamic padding
        in a token classification task) the predictions will be padded (on the right) to allow for concatenation into
        one array. The padding index is -100.

        </Tip>

        Returns: *NamedTuple* A namedtuple with the following keys:

            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return PredictionOutput(predictions=output.predictions, label_ids=output.label_ids, metrics=output.metrics, uncertainty=output.uncertainty)

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.

        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is an `datasets.Dataset`, columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.

        <Tip>

        If your predictions or labels have different sequence lengths (for instance because you're doing dynamic
        padding in a token classification task) the predictions will be padded (on the right) to allow for
        concatenation into one array. The padding index is -100.

        </Tip>

        Returns: *NamedTuple* A namedtuple with the following keys:

            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """
        self._max_length = max_length if max_length is not None else self.args.generation_max_length
        self._num_beams = num_beams if num_beams is not None else self.args.generation_num_beams
        # return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        # Redefined the original predict in Trainer to pass the uncertainty value
        return self.super_predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
    
    
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader.dataset):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        train_dataset = dataloader.dataset

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        uncertainty_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_uncertainty = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step #INPUT
            loss, logits, labels, uncertainty = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if uncertainty is not None:
                uncertainty = self._nested_gather(uncertainty)
                uncertainty_host = uncertainty if uncertainty_host is None else torch.cat((uncertainty_host, uncertainty), dim=0)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )
                if uncertainty_host is not None:
                    uncertainty = nested_numpify(uncertainty_host)
                    all_uncertainty = uncertainty if all_uncertainty is None else np.concatenate((all_uncertainty, uncertainty), axis=0)

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
        if uncertainty_host is not None:
            uncertainty = nested_numpify(uncertainty_host)
            all_uncertainty = uncertainty if all_uncertainty is None else np.concatenate((all_uncertainty, uncertainty), axis=0)
        # Number of samples
        if has_length(train_dataset):
            num_samples = len(train_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(train_dataset, IterableDatasetShard) and hasattr(train_dataset, "num_examples"):
            num_samples = train_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        if all_uncertainty is not None:
            all_uncertainty = all_uncertainty[:num_samples]
        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(dataset=train_dataset, preds=all_preds, save_prefix=metric_key_prefix)
        else:
            metrics = {}

        metrics["global_step"] = self.state.global_step

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples, uncertainty = all_uncertainty)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs #INPUT-1 --> simply put things to device
        inputs = self._prepare_inputs(inputs)
        Uncertainty = None
        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            #Modify for newer transformers
            # "max_length": self.model.config.max_length,
            # "num_beams": self.model.config.num_beams,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
        }
        if self.args.skip_generate: # Speed hack. For those uncertainty prediction with label, skip generate
            gen_kwargs['max_length'] = 1
        if self.args.uncertainty is not None:
            gen_kwargs['output_scores'] = True
            gen_kwargs['return_dict_in_generate'] = True

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        if "head_mask" in inputs:
            gen_kwargs["head_mask"] = inputs["head_mask"]
        if "decoder_head_mask" in inputs:
            gen_kwargs["decoder_head_mask"] = inputs["decoder_head_mask"]
        if "prefix_inputs" in inputs:
            with torch.no_grad():
                encoder_prefix_inputs, decoder_prefix_inputs = self.model.get_encoder_decoder_prefix_inputs(inputs['prefix_inputs'])
            gen_kwargs["encoder_prefix_inputs"] = encoder_prefix_inputs
            gen_kwargs["decoder_prefix_inputs"] = decoder_prefix_inputs
        if "decoder_BatchLabelSpaceList" in inputs:
            # print("IN", inputs["decoder_BatchLabelSpaceList"])
            gen_kwargs["decoder_BatchLabelSpaceList"] = inputs["decoder_BatchLabelSpaceList"]
            
        # prepare generation inputs
        # some encoder-decoder models can have varying encder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]
        #INPUT-2
        # print("Start Generate...")
        with torch.no_grad():
            generated_output = self.model.generate(
                generation_inputs,
                **gen_kwargs,
            )
        if self.args.uncertainty is not None:
            generated_tokens = generated_output.sequences
        else:
            generated_tokens = generated_output
        # print("End Generate...\n\n")
        # in case the batch is shorter than max length, the output should be padded
        # print("------------------------------- Generated Tokens -------------------------------")
        # print("generation_inputs size: ", generation_inputs.size())
        # print("generated tokens size: ", generated_tokens.size())
        # print("generated tokens exp1: ", generated_tokens[0])
        # print("------------------------------- End Generated Tokens -------------------------------")
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            if has_labels:
                with self.autocast_smart_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            labels = None
        # Cal Uncertainty
        with torch.no_grad():
            if self.args.uncertainty is not None:
                Uncertainty  = []
                generated_scores_flat = generated_output.scores
                generated_scores = torch.cat([token_scores.unsqueeze(1) for token_scores in generated_scores_flat], 1)
                if self.args.uncertainty == "perplexity":
                    # Calculating the perplexity towards generated tokens
                    loss_fct = CrossEntropyLoss(ignore_index=0)
                    # gen_seq_length = int(generated_scores.size(1))
                    for gen_score, gen_token in zip(generated_scores, generated_output.sequences):
                        log_perplexity = loss_fct(gen_score, gen_token[1:])
                        Uncertainty.append(torch.exp(log_perplexity))
                if "PI-" in self.args.uncertainty: # Perturb Instruction
                    # if "-HL-" in self.args.uncertainty: # has label
                    # Calculating the loss towards label
                    loss_fct = CrossEntropyLoss(ignore_index=-100)
                    with self.autocast_smart_context_manager():
                        if self.args.perturb_inst == "MC":
                            self.enable_dropout()
                        logits = (model(**inputs)["logits"] if isinstance(outputs, dict) else outputs[1]).detach()
                        if self.args.perturb_inst == "MC":
                            self.disable_dropout()
                    for logit, label in zip(logits, inputs["labels"]):
                        Uncertainty.append(loss_fct(logit, label))
                Uncertainty = torch.stack(Uncertainty)

        
        return (loss, generated_tokens, labels, Uncertainty)
    def enable_dropout(self):
        """ Function to enable the dropout layers during test-time """
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
    def disable_dropout(self):
        """ Function to enable the dropout layers during test-time """
        self.model.eval()
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if not self.args.shuffle_training_dataset:
            #Use get_eval_sampler as get_train_sampler
            if self.train_dataset is None or not has_length(self.train_dataset):
                return None
            
            # Deprecated code
            if self.args.use_legacy_prediction_loop:
                if is_torch_tpu_available():
                    return SequentialDistributedSampler(
                        self.train_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal()
                    )
                elif is_sagemaker_mp_enabled():
                    return SequentialDistributedSampler(
                        self.train_dataset,
                        num_replicas=smp.dp_size(),
                        rank=smp.dp_rank(),
                        batch_size=self.args.per_device_train_batch_size,
                    )
                elif self.args.local_rank != -1:
                    return SequentialDistributedSampler(self.train_dataset)
                else:
                    return SequentialSampler(self.train_dataset)

            if self.args.world_size <= 1:
                return SequentialSampler(self.train_dataset)
            else:
                return ShardSampler(
                    self.train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
        else:
            if self.train_dataset is None or not has_length(self.train_dataset):
                return None

            generator = None
            if self.args.world_size <= 1:
                generator = torch.Generator()
                # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
                # `args.seed`) if data_seed isn't provided.
                # Further on in this method, we default to `args.seed` instead.
                if self.args.data_seed is None:
                    seed = int(torch.empty((), dtype=torch.int64).random_().item())
                else:
                    seed = self.args.data_seed
                generator.manual_seed(seed)

            seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed

            # Build the sampler.
            if self.args.group_by_length:
                if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                    lengths = (
                        self.train_dataset[self.args.length_column_name]
                        if self.args.length_column_name in self.train_dataset.column_names
                        else None
                    )
                else:
                    lengths = None
                model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
                if self.args.world_size <= 1:
                    return LengthGroupedSampler(
                        self.args.train_batch_size * self.args.gradient_accumulation_steps,
                        dataset=self.train_dataset,
                        lengths=lengths,
                        model_input_name=model_input_name,
                        generator=generator,
                    )
                else:
                    return DistributedLengthGroupedSampler(
                        self.args.train_batch_size * self.args.gradient_accumulation_steps,
                        dataset=self.train_dataset,
                        num_replicas=self.args.world_size,
                        rank=self.args.process_index,
                        lengths=lengths,
                        model_input_name=model_input_name,
                        seed=seed,
                    )

            else:
                if self.args.world_size <= 1:
                    return RandomSampler(self.train_dataset, generator=generator)
                elif (
                    self.args.parallel_mode in [ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL]
                    and not self.args.dataloader_drop_last
                ):
                    # Use a loop for TPUs when drop_last is False to have all batches have the same size.
                    return DistributedSamplerWithLoop(
                        self.train_dataset,
                        batch_size=self.args.per_device_train_batch_size,
                        num_replicas=self.args.world_size,
                        rank=self.args.process_index,
                        seed=seed,
                    )
                else:
                    return DistributedSampler(
                        self.train_dataset,
                        num_replicas=self.args.world_size,
                        rank=self.args.process_index,
                        seed=seed,
                    )
