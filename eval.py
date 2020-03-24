import sys


def get_entities(tags):
    entities = []
    ent_type = ''
    st = -1
    for i in range(len(tags)):
        tag = tags[i]
        if '-' not in tag or tag.split('-')[1] != ent_type:
            ent_type = ''
        if tag[0] == 'E' and tag.split('-')[1] == ent_type and st != -1:
            entities.append([st, i, ent_type])
            st = -1
            ent_type = ''
        if tag[0] == 'B':
            st = i
            ent_type = tag.split('-')[1]
    return entities


def evaluate(pred_path, test_path):
    all_labels = ['O', 'B-TV', 'I-TV', 'E-TV', 'B-PER', 'I-PER', 'E-PER', 'B-NUM', 'I-NUM', 'E-NUM']
    pred_data = open(pred_path, 'r', encoding='utf-8').read().split('\n\n')
    test_data = open(test_path, 'r', encoding='utf-8').read().split('\n\n')

    pred_label, test_label = {}, {}
    pred_right_label, pred_wrong_label = {}, {}
    wrong_sample = []
    miss_sample = []
    pred_num, pred_right_num, true_num = 0, 0, 0

    for i in range(len(test_data)):
        if test_data[i] == '' or test_data[i][0] == '\n':
            break
        pred_sent, test_sent = pred_data[i].split('\n'), test_data[i].split('\n')
        if pred_sent[-1] == '':
            pred_sent.pop(-1)
        if test_sent[-1] == '':
            test_sent.pop(-1)

        pred_tags = [p.split(' ')[-1] for p in pred_sent]
        test_tags = [t.split(' ')[-1] for t in test_sent]

        for tag in pred_tags:
            if tag not in all_labels:
                print('Warning, there are a label should not exist: {}'.format(tag))
                sys.exit(0)

        sent = [p[0] for p in pred_sent]
        assert len(pred_tags) == len(test_tags)

        pred_entities = get_entities(pred_tags)
        test_entities = get_entities(test_tags)
        pred_num += len(pred_entities)
        true_num += len(test_entities)
        pred_right_num += sum([1 if ent in test_entities else 0 for ent in pred_entities])

        flag = 0
        sent_ = sent[:]
        for ent in pred_entities:
            pred_label[ent[2]] = pred_label[ent[2]] + 1 if ent[2] in pred_label.keys() else 1
            if ent in test_entities:
                pred_right_label[ent[2]] = pred_right_label[ent[2]] + 1 if ent[2] in pred_right_label.keys() else 1
                sent_[ent[0]] = '《({})'.format(ent[2]) + sent_[ent[0]]
                sent_[ent[1]] = sent_[ent[1]] + '》'
            else:
                pred_wrong_label[ent[2]] = pred_wrong_label[ent[2]] + 1 if ent[2] in pred_wrong_label.keys() else 1
                sent_[ent[0]] = '《({})*'.format(ent[2]) + sent_[ent[0]]
                sent_[ent[1]] = sent_[ent[1]] + '*》'
                flag = 1
        if flag:
            wrong_sample.append(''.join(sent_))
            for ent in test_entities:
                sent[ent[0]] = '《({})'.format(ent[2]) + sent[ent[0]]
                sent[ent[1]] = sent[ent[1]] + '》'
            wrong_sample.append(''.join(sent))
            flag = 0

        sent = [p.split(' ')[0] for p in pred_sent]
        for ent in test_entities:
            test_label[ent[2]] = test_label[ent[2]] + 1 if ent[2] in test_label.keys() else 1
            if ent in pred_entities:
                sent[ent[0]] = '《({})'.format(ent[2]) + sent[ent[0]]
                sent[ent[1]] = sent[ent[1]] + '》'
            else:
                sent[ent[0]] = '《({})#'.format(ent[2]) + sent[ent[0]]
                sent[ent[1]] = sent[ent[1]] + '#》'
                flag = 1
        if flag:
            miss_sample.append(''.join(sent))

    pre = pred_right_num / pred_num if pred_num > 0 else 0
    rec = pred_right_num / true_num if true_num > 0 else 0
    f = (2 * pre * rec) / (pre + rec)
    return pre, rec, f, pred_label, test_label, pred_right_label, pred_wrong_label, wrong_sample, miss_sample


if __name__ == '__main__':
    pred_path = sys.argv[1]
    test_path = sys.argv[2]

    precision, recall, f1, pred_label, test_label, pred_right_label, pred_wrong_label, wrong_sample, miss_sample \
        = evaluate(pred_path, test_path)
    print('Precision:{:.2f}, recall:{:.2f}, f1:{:.2f}'.format(precision * 100, recall * 100, f1 * 100))
    for ent_type in pred_label:
        print('Type {} predict {} entities, {} of it is right,and the gold data has {} entities in this type.'.format(
            ent_type, pred_label[ent_type],pred_right_label[ent_type] if ent_type in pred_right_label.keys() else 0,
            test_label[ent_type] if ent_type in test_label.keys() else 0))
    
        
    file = open('Wrong_sample.txt', 'w', encoding='utf-8')
    file.write('In this sample text, the words in 《   》 means the entity you predict right.\n')
    file.write('The words in 《*   *》 means the entity you predict wrong.\n')
    file.write('##############################################################################\n')
    for idx in range(len(wrong_sample)):
        sent = wrong_sample[idx]
        if idx % 2 == 0:
            file.write('The wrong sample:  ' + sent + '\n')
        else:
            file.write('The gold sample:  ' + sent + '\n\n')
    file.close()
    print('The wrong sample have been saved as Wrong_sample.txt')
        
    file = open('Missed_sample.txt', 'w', encoding='utf-8')
    file.write('In this sample text, the words in 《   》 means the entity you predict right.\n')
    file.write('The words in 《#   #》 means the entity you missed.\n')
    file.write('##############################################################################\n')
    for sent in miss_sample:
        file.write(sent + '\n\n')
    file.close()
    print('The missing sample have been saved as Missed_sample.txt')
    


