"""ILPO network for vectors."""
from ilpo import ILPO
from utils import *

'''
这里应该就是要改的全部了

'''
class CatchILPO(ILPO):


    def process_inputs(self, inputs):
        inputs = np.array(inputs) / args.max_x
        return inputs.tolist()

    def load_examples(self):
        if args.input_dir is None or not os.path.exists(args.input_dir):
            raise Exception("input_dir does not exist")
        input_paths = glob.glob(os.path.join(args.input_dir, "*.dat"))
        if len(input_paths) == 0:
            raise Exception("input_dir contains no image files")

        def get_name(path):
            name, _ = os.path.splitext(os.path.basename(path))
            return name

        if all(get_name(path).isdigit() for path in input_paths):
            input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
        else:
            input_paths = sorted(input_paths)
            # 用图片输入明显不合适来了
        with tf.name_scope("load_datas"):
            paths = []
            inputs = []
            data_a = []
            data_b = []
            data_c=[]
            for demonstration in input_paths:
                temp = []
                with open(demonstration) as f:
                    for trajectory in f:
                        s = trajectory.strip().split(' ')
                        temp.append(list(map(int, s)))
                        if len(temp) is args.height:
                            temp=self.process_inputs(temp)
                            inputs.append(temp)
                            temp = []
                            if len(inputs) > 1:
                                data_a.append(inputs[-2])
                                data_b.append(extract_action(inputs[-1]))
                                paths.append(str([data_a[-1],data_b[-1]]))
                                data_c.append(extract_action(inputs[-2]))
                assert len(temp) is 0
                inputs = []
        num_samples = len(paths)

        #data_a=np.reshape(data_a,[np.shape(data_a)[0],args.height,args.width])
        #data_b=np.reshape(data_b, [np.shape(data_b)[0], -1])

        inputs = tf.convert_to_tensor(data_a, tf.float32)
        #inputs=self.process_inputs(inputs)
        targets = tf.convert_to_tensor(data_b, tf.float32)
        input_actions=tf.convert_to_tensor(data_c, tf.float32)
        paths_batch, inputs_batch, targets_batch, action_batch = tf.train.shuffle_batch(
            [paths, inputs, targets,input_actions],
            batch_size=args.batch_size,
            num_threads=1,
            enqueue_many=True,
            capacity=num_samples,
            min_after_dequeue=10)

        inputs_batch.set_shape([args.batch_size, args.height,args.width])
        targets_batch.set_shape([args.batch_size, targets_batch.shape[-1]])
        action_batch.set_shape([args.batch_size, targets_batch.shape[-1]])
        steps_per_epoch = int(math.ceil(num_samples / args.batch_size))

        return Examples(
            paths=paths_batch,
            inputs=inputs_batch,
            targets=targets_batch,
            input_actions=action_batch,
            count=len(input_paths),
            steps_per_epoch=steps_per_epoch,
        )

    def load_test(self):
        if args.test_dir is None or not os.path.exists(args.test_dir):
            raise Exception("input_dir does not exist")
        input_paths = glob.glob(os.path.join(args.test_dir, "*.dat"))
        if len(input_paths) == 0:
            raise Exception("input_dir contains no image files")

        def get_name(path):
            name, _ = os.path.splitext(os.path.basename(path))
            return name

        if all(get_name(path).isdigit() for path in input_paths):
            input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
        else:
            input_paths = sorted(input_paths)
            # 用图片输入明显不合适来了
        with tf.name_scope("load_datas"):
            paths = []
            inputs = []
            data_a = []
            data_b = []
            data_c = []
            for demonstration in input_paths:
                temp = []
                with open(demonstration) as f:
                    for trajectory in f:
                        s = trajectory.strip().split(' ')
                        temp.append(list(map(int, s)))
                        if len(temp) is args.height:
                            temp=self.process_inputs(temp)
                            inputs.append(temp)
                            temp = []
                            if len(inputs) > 1:
                                data_a.append(inputs[-2])
                                data_b.append(extract_action(inputs[-1]))
                                paths.append(str([data_a[-1], data_b[-1]]))
                                data_c.append(extract_action(inputs[-2]))
                assert len(temp) is 0
                inputs = []
        num_samples = len(paths)

        # data_a=np.reshape(data_a,[np.shape(data_a)[0],args.height,args.width])
        # data_b=np.reshape(data_b, [np.shape(data_b)[0], -1])

        inputs = tf.convert_to_tensor(data_a, tf.float32)
        #inputs = self.process_inputs(inputs)
        targets = tf.convert_to_tensor(data_b, tf.float32)
        input_actions = tf.convert_to_tensor(data_c, tf.float32)
        paths_batch, inputs_batch, targets_batch, action_batch = tf.train.shuffle_batch(
            [paths, inputs, targets, input_actions],
            batch_size=args.batch_size,
            num_threads=1,
            enqueue_many=True,
            capacity=num_samples,
            min_after_dequeue=10)

        inputs_batch.set_shape([args.batch_size, args.height, args.width])
        targets_batch.set_shape([args.batch_size, targets_batch.shape[-1]])
        action_batch.set_shape([args.batch_size, targets_batch.shape[-1]])
        steps_per_epoch = int(math.ceil(num_samples / args.batch_size))

        return Examples(
            paths=paths_batch,
            inputs=inputs_batch,
            targets=targets_batch,
            input_actions=action_batch,
            count=len(input_paths),
            steps_per_epoch=steps_per_epoch,
        )

    def create_encoder(self, state):
        """Creates state embedding."""

        layers = []

        with tf.variable_scope("encoder_1"):
            output = fully_connected(state, args.ngf)
            layers.append(output)

        layer_specs = [
            args.ngf * 2,
        ]

        for out_channels in layer_specs:
            with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
                rectified = lrelu(layers[-1], 0.2)
                encoded = fully_connected(rectified, out_channels)
                layers.append(encoded)

        return layers

    def create_generator(self, layers, generator_outputs_channels):
        """Returns next state prediction given state and latent action."""

        s_t_layers = list(layers)

        with tf.variable_scope("decoder_1"):
            inp = s_t_layers[-1]
            rectified = lrelu(inp, 0.2)
            output = fully_connected(rectified, args.ngf)
            s_t_layers.append(output)

        with tf.variable_scope("decoder_2"):
            inp = s_t_layers[-1]
            rectified = lrelu(inp, 0.2)
            output = fully_connected(rectified, generator_outputs_channels)
            s_t_layers.append(output)

        return s_t_layers[-1]

    def train_examples(self, examples,test):
        print("examples count = %d" % examples.count)
        print("test count= %d" % test.count)
        model = self.create_model(examples.inputs, examples.targets,examples.input_actions, test.inputs, test.targets,test.input_actions)
        tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name + "/values", var)

        for grad, var in model.gen_grads_and_vars:
            tf.summary.histogram(var.op.name + "/gradients", grad)

        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])
        with tf.name_scope("encode_vectors"):
            display_fetches = {
                "paths": examples.paths,
                "inputs": examples.inputs,
                "targets": examples.targets,
                "outputs": model.outputs
            }
        #这里我不太确定对不对
        saver = tf.train.Saver(max_to_keep=1)

        logdir = args.output_dir if (args.trace_freq > 0 or args.summary_freq > 0) else None
        sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.2)
        #TODO:在自己电脑上测试时应该改一下

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with sv.managed_session(config=config) as sess:
            print("parameter_count =", sess.run(parameter_count))
            if args.checkpoint is not None:
                print("loading model from checkpoint")
                checkpoint = tf.train.latest_checkpoint(args.checkpoint)
                saver.restore(sess, checkpoint)

            max_steps = 2 ** 32
            #这么大的吗
            if args.max_epochs is not None:
                max_steps = examples.steps_per_epoch * args.max_epochs
            if args.max_steps is not None:
                max_steps = args.max_steps

            start = time.time()
            I=0
            train_loss=0.0
            train_acc=0.0
            test_acc=0.0
            for step in range(max_steps):
                I+=1
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = None
                if should(args.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(args.progress_freq):
                    fetches["gen_loss_L1"] = model.gen_loss_L1


                if should(args.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(args.display_freq):

                    fetches["display"] = display_fetches

                results = sess.run(fetches, options=options, run_metadata=run_metadata)
                train_acc+=sess.run(model.test_acc)
                train_loss+=sess.run(model.gen_loss_L1)
                test_acc+=sess.run(model.test_acc)
                if should(args.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(args.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * args.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * args.batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (
                    train_epoch, train_step, rate, remaining / 60))
                    print("gen_loss_L1", train_loss/I)
                    print("train_acc", train_acc/I)
                    print("test_acc",test_acc/I)
                    I=0
                    train_loss=0
                    train_acc=0
                    test_acc=0
                if should(args.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(args.output_dir, "model"), global_step=sv.global_step)
                if sv.should_stop():
                    break



def main():
    model = CatchILPO()
    model.run()

if __name__ == "__main__":
    main()
