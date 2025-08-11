import pyterrier as pt
import pyterrier_rag

from pyterrier_t5 import MonoT5ReRanker

def main():
    # initialize bm25,monot5
    sparse_index = pt.Artifact.from_hf('pyterrier/ragwiki-terrier')

    # 这里管道需要进一步研究与注释
    bm25_ret = pt.rewrite.tokenise() >> sparse_index.bm25(include_fields=['docno', 'text', 'title'], threads=5) >> pt.rewrite.reset()
    monoT5 = MonoT5ReRanker(batch_size=64, verbose=False)
    monoT5_ret = bm25_ret%10 >> monoT5

    #initialize fid
    fid = pyterrier_rag.readers.T5FiD("terrierteam/t5fid_base_nq")

    #build pipline
    user_text = input("Please Type Your Question Here:")
    monoT5_fid_mp = bm25_ret%200 >> monoT5%100 >> fid
    qanswer_frame = monoT5_fid_mp.search(user_text)

    print(qanswer_frame['qanswer'].iloc[0])

if __name__ == "__main__":
    main()