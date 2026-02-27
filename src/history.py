
import os
import sqlite3
import json
import uuid
import shutil
import datetime
import logging
from typing import Dict, Any, List, Optional
from src.state import AgentState

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HistoryManager:
    def __init__(self, db_path: str = "history_data/history.db", data_dir: str = "history_data"):
        """
        Initialize the HistoryManager with SQLite database and storage directory.
        """
        self.data_dir = data_dir
        self.db_path = db_path
        self.images_dir = os.path.join(self.data_dir, "images")
        
        # Ensure directories exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
            
        self._init_db()

    def _init_db(self):
        """Create the history table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table for history
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_history (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            title TEXT,
            source_type TEXT,
            source_name TEXT,
            summary TEXT,
            keywords TEXT,
            state_json TEXT NOT NULL
        )
        ''')
        
        conn.commit()
        conn.close()

    def save_analysis(self, state: AgentState) -> str:
        """
        Save the analysis result (AgentState) to local history.
        Returns the unique ID of the saved record.
        """
        try:
            # Generate unique ID and timestamp
            record_id = str(uuid.uuid4())
            timestamp = datetime.datetime.now().isoformat()
            
            # Extract metadata
            metadata = state.get("metadata", {})
            title = metadata.get("Title") or metadata.get("title") or "Untitled Analysis"
            source_name = os.path.basename(state.get("source", "Unknown Source"))
            source_type = "PDF" if source_name.lower().endswith(".pdf") else "Arxiv"
            
            # Extract summary (first 200 chars of report or translation)
            report = state.get("final_report", "")
            summary = report[:200] + "..." if report else "No summary available."
            
            # Handle images: Copy from temp to persistent storage
            original_figures = state.get("figures", [])
            new_figures_paths = []
            
            if original_figures:
                record_images_dir = os.path.join(self.images_dir, record_id)
                if not os.path.exists(record_images_dir):
                    os.makedirs(record_images_dir)
                
                for img_path in original_figures:
                    if os.path.exists(img_path):
                        # Copy image
                        filename = os.path.basename(img_path)
                        new_path = os.path.join(record_images_dir, filename)
                        shutil.copy2(img_path, new_path)
                        new_figures_paths.append(new_path)
            
            # Update state with new image paths for persistence
            # Create a copy to avoid modifying the runtime state in-place if not desired
            # But here we want to save the persistent paths
            persistent_state = state.copy()
            persistent_state["figures"] = new_figures_paths
            
            # Serialize state
            state_json = json.dumps(persistent_state, ensure_ascii=False)
            
            # Keywords (placeholder for now, could be extracted from report)
            keywords = "Analysis, Paper" 
            
            # Insert into DB
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO analysis_history (id, timestamp, title, source_type, source_name, summary, keywords, state_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (record_id, timestamp, title, source_type, source_name, summary, keywords, state_json))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Saved analysis history: {record_id}")
            return record_id
            
        except Exception as e:
            logger.error(f"Failed to save analysis history: {e}")
            raise e

    def get_all_history(self, sort_by: str = "timestamp", order: str = "DESC") -> List[Dict[str, Any]]:
        """
        Retrieve all history records (metadata only) for list view.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row # Return dict-like rows
        cursor = conn.cursor()
        
        query = f"SELECT id, timestamp, title, source_type, source_name, summary, keywords FROM analysis_history ORDER BY {sort_by} {order}"
        cursor.execute(query)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]

    def search_history(self, query: str) -> List[Dict[str, Any]]:
        """
        Search history by title, keywords, or source name.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        search_term = f"%{query}%"
        sql = '''
        SELECT id, timestamp, title, source_type, source_name, summary, keywords 
        FROM analysis_history 
        WHERE title LIKE ? OR keywords LIKE ? OR source_name LIKE ?
        ORDER BY timestamp DESC
        '''
        cursor.execute(sql, (search_term, search_term, search_term))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]

    def get_analysis_by_id(self, record_id: str) -> Optional[AgentState]:
        """
        Retrieve the full analysis state by ID.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT state_json FROM analysis_history WHERE id = ?", (record_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            try:
                state = json.loads(row[0])
                return state
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON for record {record_id}")
                return None
        return None

    def delete_analysis(self, record_id: str) -> bool:
        """
        Delete a history record and its associated files.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if exists
            cursor.execute("SELECT id FROM analysis_history WHERE id = ?", (record_id,))
            if not cursor.fetchone():
                conn.close()
                return False
            
            # Delete from DB
            cursor.execute("DELETE FROM analysis_history WHERE id = ?", (record_id,))
            conn.commit()
            conn.close()
            
            # Delete images directory
            record_images_dir = os.path.join(self.images_dir, record_id)
            if os.path.exists(record_images_dir):
                shutil.rmtree(record_images_dir)
                
            logger.info(f"Deleted analysis history: {record_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete analysis history: {e}")
            return False

    def export_history_to_file(self, record_id: str, format: str = "json") -> str:
        """
        Export a specific history record to a file. 
        Returns the file path.
        """
        state = self.get_analysis_by_id(record_id)
        if not state:
            raise ValueError("Record not found")
            
        export_dir = os.path.join(self.data_dir, "exports")
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
            
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"export_{record_id}_{timestamp_str}.{format}"
        file_path = os.path.join(export_dir, filename)
        
        if format == "json":
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        elif format == "md":
            # Combine all markdown content
            content = f"# {state.get('metadata', {}).get('Title', 'Analysis Report')}\n\n"
            content += f"## Report\n{state.get('final_report', '')}\n\n"
            content += f"## Translation\n{state.get('translation', '')}\n\n"
            # Add other sections as needed
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        return file_path
