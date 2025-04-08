from autogen.agentchat.contrib.capabilities.teachability import Teachability, MemoStore
from termcolor import colored

class DedupMemoStore(MemoStore):
    # def __init__(self, similarity_threshold: float = 0.3, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.similarity_threshold = similarity_threshold
    #     self._clean_and_reindex()


    def __init__(self, similarity_threshold: float = 0.3, *args, **kwargs):
        if 'reset' in kwargs and kwargs['reset']:
            self.using_cosine = True
        else:
            self.using_cosine = True
            
        super().__init__(*args, **kwargs)        
        if self.using_cosine:
            self.db_client.delete_collection("memos")
            self.vec_db = self.db_client.create_collection(
                "memos",
                metadata={"hnsw:space": "cosine"}
            )
            
        self.similarity_threshold = similarity_threshold
        self._clean_and_reindex()


    def _clean_and_reindex(self):
        """Clean up missing entries and reindex the remaining ones"""
        # Get valid entries
        valid_entries = []
        for uid, (input_text, output_text) in self.uid_text_dict.items():
            try:
                # Verify if embedding exists
                self.vec_db.query(query_texts=[""], n_results=1, filter={"id": uid})
                valid_entries.append((input_text, output_text))
            except:
                continue

        if not valid_entries:
            return

        # Clear existing data
        self.uid_text_dict.clear()
        self.vec_db.delete(delete_all=True)

        # Reinsert with new sequential IDs
        for input_text, output_text in valid_entries:
            super().add_input_output_pair(input_text, output_text)

    def is_duplicate(self, input_text: str, output_text: str) -> bool:
        if len(self.uid_text_dict) == 0:
            return False
            
        # Check for exact matches first
        for stored_input, stored_output in self.uid_text_dict.values():
            if input_text == stored_input and output_text == stored_output:
                if self.verbosity >= 1:
                    print(colored("\nEXACT DUPLICATE DETECTED - SKIPPING", "yellow"))
                return True

        # Check for semantic similarity
        try:
            results = self.vec_db.query(
                query_texts=[input_text],
                n_results=1
            )
            
            if not results["ids"][0]:
                return False
                
            closest_distance = results["distances"][0][0]
            closest_id = results["ids"][0][0]
            closest_input, closest_output = self.uid_text_dict[closest_id]
            
            if closest_distance < self.similarity_threshold:
                if self.verbosity >= 1:
                    print(colored(
                        f"\nSIMILAR MEMORY DETECTED (distance: {closest_distance})\n" +
                        f"Existing:\n  Input: {closest_input}\n  Output: {closest_output}\n" +
                        f"New:\n  Input: {input_text}\n  Output: {output_text}",
                        "yellow"
                    ))
                return True
        except Exception as e:
            print(colored(f"\nError checking similarity: {str(e)}", "red"))
            
        return False

    def add_input_output_pair(self, input_text: str, output_text: str):
        if self.is_duplicate(input_text, output_text):
            if self.verbosity >= 1:
                print(colored("\nSkipping duplicate memory", "yellow"))
            return False
        success = super().add_input_output_pair(input_text, output_text)
        if success:
            self._clean_and_reindex()
        return success

    def get_related_memos(self, query_text: str, n_results: int = None, **kwargs) -> list:
        try:
            # Ensure n_results is at least 1
            if not n_results or n_results < 1:
                n_results = 1
                
            results = self.vec_db.query(
                query_texts=[query_text],
                n_results=n_results
            )
            
            memo_list = []
            for i, uid in enumerate(results["ids"][0]):
                try:
                    input_text, output_text = self.uid_text_dict[uid]
                    memo_list.append((input_text, output_text))
                except KeyError:
                    # This shouldn't happen now with reindexing
                    print(colored(f"\nWarning: Could not find memo with ID {uid}", "yellow"))
                    continue
                    
            return memo_list
            
        except Exception as e:
            print(colored(f"\nError in get_related_memos: {str(e)}", "red"))
            return []

# class DedupTeachability(Teachability):
#     def __init__(self, similarity_threshold: float = 0.3, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.memo_store = DedupMemoStore(
#             similarity_threshold=similarity_threshold,
#             verbosity=self.verbosity,
#             path_to_db_dir=self.path_to_db_dir
#         )

# class DedupTeachability(Teachability):
#     def __init__(self, similarity_threshold: float = 0.3, *args, **kwargs):
#         # Patch the colored function to handle 'light_green'
#         import functools
#         original_colored = colored
        
#         @functools.wraps(original_colored)
#         def patched_colored(text, color, *args, **kwargs):
#             if color == 'light_green':
#                 color = 'green'  # Replace with a valid color
#             return original_colored(text, color, *args, **kwargs)
        
#         # Monkey patch the colored function in the teachability module
#         import autogen.agentchat.contrib.capabilities.teachability
#         autogen.agentchat.contrib.capabilities.teachability.colored = patched_colored
        
#         # Now initialize normally
#         super().__init__(*args, **kwargs)
#         self.memo_store = DedupMemoStore(
#             similarity_threshold=similarity_threshold,
#             verbosity=self.verbosity,
#             path_to_db_dir=self.path_to_db_dir
#         )

class DedupTeachability(Teachability):
    def __init__(self, similarity_threshold: float = 0.3, use_cosine: bool = True, *args, **kwargs):
        import functools
        original_colored = colored
        
        @functools.wraps(original_colored)
        def patched_colored(text, color, *args, **kwargs):
            if color == 'light_green':
                color = 'green' 
            return original_colored(text, color, *args, **kwargs)
        
        import autogen.agentchat.contrib.capabilities.teachability
        autogen.agentchat.contrib.capabilities.teachability.colored = patched_colored
        
        super().__init__(*args, **kwargs)
        
        # MemoStore with cosine similarity
        self.memo_store = DedupMemoStore(
            similarity_threshold=similarity_threshold,
            verbosity=self.verbosity,
            path_to_db_dir=self.path_to_db_dir,
            reset=kwargs.get('reset', False)  
        )
        
        if use_cosine and hasattr(self.memo_store, 'db_client') and hasattr(self.memo_store, 'vec_db'):
            try:
                current_collection = self.memo_store.vec_db
                
                collection_info = self.memo_store.db_client.get_collection("memos")
                metadata = getattr(collection_info, 'metadata', {}) or {}
                
                if metadata.get("hnsw:space") != "cosine":
                    self.memo_store.db_client.delete_collection("memos")
                    self.memo_store.vec_db = self.memo_store.db_client.create_collection(
                        "memos",
                        metadata={"hnsw:space": "cosine"}
                    )
                    if hasattr(self.memo_store, '_clean_and_reindex'):
                        self.memo_store._clean_and_reindex()
            except Exception as e:
                print(f"Error setting up cosine similarity: {e}")