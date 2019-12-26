import tensorflow as tf



def loss_layer(output, target_labels, target_scores, target_loc, threshold=0.5):


    predictions_loc, predictions_score = output


    dtype = predictions_loc[0].dtype

    l_cross_pos = []

    l_cross_neg = []

    l_loc = []


    for i in range(len(predictions_score)):

        pred_loc = predictions_loc[i]

        pred_score = predictions_score[i]

        true_label = tf.cast(target_labels[i], tf.int32)


        pos_mask = target_scores[i] > threshold

        no_classes = tf.cast(pos_mask, tf.int32)

        fpos_mask = tf.cast(pos_mask, dtype)


        pos_num = tf.reduce_sum(fpos_mask)


        neg_mask = tf.logical_not(pos_mask)

        fneg_mask = tf.cast(neg_mask, dtype)


        neg_values = tf.where(

            neg_mask, pred_score[:, 0], 1.-fneg_mask)


        neg_values_flat = tf.reshape(neg_values, [-1])


        n_neg = tf.cast(3 * pos_num, tf.int32)


        n_neg = tf.maximum(n_neg, tf.size(neg_values_flat) // 8)


        n_neg = tf.maximum(n_neg, tf.shape(neg_values)[0] * 4)


        max_neg_entries = tf.cast(tf.reduce_sum(fneg_mask), tf.int32)


        n_neg = tf.minimum(n_neg, max_neg_entries)


        val, idxes = tf.nn.top_k(-neg_values_flat, k=n_neg)


        minval = val[-1]


        neg_mask = tf.logical_and(neg_mask, -neg_values > minval)


        fneg_mask = tf.cast(neg_mask, dtype)


        with tf.name_scope('cross_entropy_pos'):


            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(

                logits=pred_score, labels=true_label

            )


            loss = tf.losses.compute_weighted_loss(loss, fpos_mask)


            l_cross_pos.append(loss)


        with tf.name_scope('cross_entropy_neg'):


            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(

                logits=pred_score[:, :2], labels=no_classes

            )


            loss = tf.losses.compute_weighted_loss(loss, fneg_mask)


            l_cross_neg.append(loss)


        with tf.name_scope('localization'):


            weights = tf.expand_dims(fpos_mask, axis=-1)


            loss = abs_smooth(

                pred_loc - target_loc[i])


            loss = tf.losses.compute_weighted_loss(loss, weights)


            l_loc.append(loss)


    with tf.name_scope('total'):


        l_cross_pos = tf.gather(

            l_cross_pos, tf.where(tf.not_equal(l_cross_pos, 0))

        )


        l_cross_neg = tf.gather(

            l_cross_neg, tf.where(tf.not_equal(l_cross_neg, 0))

        )


        l_loc = tf.gather(

            l_loc, tf.where(tf.not_equal(l_loc, 0))

        )


        total_cross_pos = tf.reduce_mean(l_cross_pos)


        total_cross_neg = tf.reduce_mean(l_cross_neg)


        total_loc = tf.reduce_mean(l_loc)


    return total_cross_pos, total_cross_neg, total_loc



def abs_smooth(x):


    absx = tf.abs(x)


    minx = tf.minimum(absx, 1)


    r = 0.5 * ((absx - 1) * minx + absx)


    return r
